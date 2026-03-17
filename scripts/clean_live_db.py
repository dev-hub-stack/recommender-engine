"""
clean_live_db.py
Applies phone normalization, city title-casing and email lowercasing to the
live mastergroup_recommendations PostgreSQL database.

Optimised for remote (RDS) execution:
  - Pre-loads the full set of customer_statistics IDs into memory (one query).
  - Accumulates all UPDATE/DELETE statements per batch then flushes in a single
    executemany call.
  - Zero per-row network round-trips inside the batch loop.
"""

import re
import os
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from dotenv import load_dotenv

load_dotenv('.env')

PG_HOST = os.getenv('PROD_PG_HOST', os.getenv('PG_HOST', 'localhost'))
PG_PORT = os.getenv('PROD_PG_PORT', os.getenv('PG_PORT', '5432'))
PG_DB   = os.getenv('PROD_PG_DB',  os.getenv('PG_DB',   'mastergroup_recommendations'))
PG_USER = os.getenv('PROD_PG_USER', os.getenv('PG_USER', 'postgres'))
PG_PASS = os.getenv('PROD_PG_PASSWORD', os.getenv('PG_PASSWORD', 'postgres'))

BATCH = 5000

# ---------------------------------------------------------------------------
# Normalisation helpers  (identical logic to clean_historical_data.py)
# ---------------------------------------------------------------------------

def clean_phone(val):
    if val is None:
        return None
    digits = re.sub(r'[^\d]', '', str(val).strip())
    if not digits:
        return None
    if len(digits) == 10 and digits.startswith('3'):
        return '+92' + digits
    if len(digits) == 11 and digits.startswith('03'):
        return '+92' + digits[1:]
    if len(digits) == 12 and digits.startswith('923'):
        return '+' + digits
    if len(digits) == 12 and digits.startswith('9203'):
        return '+923' + digits[4:]
    return None


def clean_city(val):
    if val is None:
        return None
    s = str(val).strip()
    return s.title() if s.lower() not in ('nan', 'none', '') else None


def clean_email(val):
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s not in ('nan', 'none', '') else None


def make_uid(phone, name):
    """Reconstruct the synthetic unified_customer_id used by the system."""
    if not phone:
        return None
    if name and str(name).strip():
        first = re.sub(r'[^a-z0-9]', '', str(name).strip().split()[0].lower())
        return f"{phone}_{first}" if first else phone
    return phone


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def execute_cleaning():
    print(f"Connecting to {PG_DB} at {PG_HOST} …", flush=True)
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT,
                             dbname=PG_DB, user=PG_USER, password=PG_PASS)
    conn.autocommit = False
    cur  = conn.cursor(cursor_factory=RealDictCursor)

    # ------------------------------------------------------------------
    # Step 1 – Pre-load all existing customer_statistics IDs into a set
    # ------------------------------------------------------------------
    print("Pre-loading customer_statistics IDs …", flush=True)
    cur.execute("SELECT customer_id FROM customer_statistics")
    existing_stat_ids = {r['customer_id'] for r in cur.fetchall()}
    print(f"  → {len(existing_stat_ids):,} stat records loaded.", flush=True)

    # ------------------------------------------------------------------
    # Step 2 – Process orders in batches
    # ------------------------------------------------------------------
    print("\n--- Processing orders table ---", flush=True)
    total_updated = 0
    offset = 0

    while True:
        cur.execute(
            "SELECT id, customer_phone, customer_city, customer_email, "
            "       customer_name, unified_customer_id "
            "FROM orders ORDER BY id "
            "LIMIT %s OFFSET %s",
            (BATCH, offset)
        )
        rows = cur.fetchall()
        if not rows:
            break

        order_updates   = []   # (phone, city, email, new_uid, id)
        stat_renames    = []   # (new_uid, old_uid)  — safe rename
        stat_merges     = []   # (new_uid, old_uid)  — need merge+delete

        for r in rows:
            orig_phone  = r['customer_phone']
            orig_city   = r['customer_city']
            orig_email  = r['customer_email']
            orig_uid    = r['unified_customer_id']

            new_phone = clean_phone(orig_phone)
            new_city  = clean_city(orig_city)
            new_email = clean_email(orig_email)

            # Only change the phone if we could parse it to a valid number
            eff_phone = new_phone if new_phone else orig_phone
            eff_city  = new_city  if new_city  else orig_city
            eff_email = new_email if new_email else orig_email

            changed = (eff_phone != orig_phone or
                       eff_city  != orig_city  or
                       eff_email != orig_email)

            new_uid = orig_uid
            uid_changed = False
            if new_phone and new_phone != orig_phone:
                new_uid = make_uid(new_phone, r['customer_name'])
                uid_changed = (new_uid != orig_uid)

            if changed or uid_changed:
                order_updates.append((eff_phone, eff_city, eff_email, new_uid, r['id']))

                if uid_changed and orig_uid:
                    if new_uid in existing_stat_ids:
                        stat_merges.append((new_uid, orig_uid))
                    elif orig_uid in existing_stat_ids:
                        stat_renames.append((new_uid, orig_uid))
                        existing_stat_ids.discard(orig_uid)
                        existing_stat_ids.add(new_uid)

        # Flush order updates
        if order_updates:
            execute_batch(cur,
                "UPDATE orders "
                "SET customer_phone=%s, customer_city=%s, customer_email=%s, "
                "    unified_customer_id=%s "
                "WHERE id=%s",
                order_updates)
            total_updated += len(order_updates)

        # Safe renames
        if stat_renames:
            execute_batch(cur,
                "UPDATE customer_statistics SET customer_id=%s WHERE customer_id=%s",
                stat_renames)

        # Merges (aggregate then delete old)
        for new_uid, old_uid in stat_merges:
            cur.execute("""
                UPDATE customer_statistics tgt
                SET total_orders     = tgt.total_orders     + src.total_orders,
                    total_products   = tgt.total_products   + src.total_products,
                    unique_products  = tgt.unique_products  + src.unique_products,
                    total_spent      = tgt.total_spent      + src.total_spent,
                    avg_order_value  = (tgt.total_spent + src.total_spent)
                                       / NULLIF(tgt.total_orders + src.total_orders, 0),
                    first_order_date = LEAST(tgt.first_order_date, src.first_order_date),
                    last_order_date  = GREATEST(tgt.last_order_date, src.last_order_date),
                    updated_at       = NOW()
                FROM customer_statistics src
                WHERE tgt.customer_id = %s AND src.customer_id = %s
            """, (new_uid, old_uid))
            cur.execute("DELETE FROM customer_statistics WHERE customer_id=%s", (old_uid,))
            existing_stat_ids.discard(old_uid)

        conn.commit()
        offset += BATCH
        print(f"  Offset {offset:,} | total updated so far: {total_updated:,}", flush=True)

    print(f"\nFinished. {total_updated:,} orders updated.", flush=True)

    # ------------------------------------------------------------------
    # Step 3 – Fix remaining city names in customer_statistics
    # ------------------------------------------------------------------
    print("\n--- Fixing city names in customer_statistics ---", flush=True)
    cur.execute("SELECT customer_id, customer_city FROM customer_statistics "
                "WHERE customer_city IS NOT NULL")
    stat_city_updates = [
        (clean_city(r['customer_city']), r['customer_id'])
        for r in cur.fetchall()
        if clean_city(r['customer_city']) != r['customer_city']
    ]
    if stat_city_updates:
        execute_batch(cur,
            "UPDATE customer_statistics SET customer_city=%s WHERE customer_id=%s",
            stat_city_updates)
        conn.commit()
    print(f"  {len(stat_city_updates):,} city values updated.", flush=True)

    print("\n✅  All done!", flush=True)
    cur.close()
    conn.close()


if __name__ == "__main__":
    execute_cleaning()
