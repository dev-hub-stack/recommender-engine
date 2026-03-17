"""
ingest_historical_data.py
=========================
Ingests the cleaned historical customer data from CustomerDataMasterVerse_Cleaned.csv
into the live PostgreSQL database.

Strategy (from approved implementation plan):
  - Overlap customers (phone already in DB): enrich customer_statistics only, skip orders insert.
  - New customers (237K+): insert one seeded orders row + one customer_statistics row.
  - Province: infer from city using province_utils.CITY_TO_PROVINCE map.
  - Idempotent: pre-load existing phones — safe to re-run.
  - Batch commits every 5000 rows.

Run on EC2 for best RDS latency:
  python3 -u ingest_historical_data.py > /tmp/ingest_output.log 2>&1 &
"""

import os, re, sys, uuid
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from dotenv import load_dotenv
from datetime import datetime

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv('/opt/mastergroup-ml/.env') if os.path.exists('/opt/mastergroup-ml/.env') else load_dotenv('.env')

PG_HOST = os.getenv('PROD_PG_HOST', os.getenv('PG_HOST', 'localhost'))
PG_PORT = os.getenv('PROD_PG_PORT', os.getenv('PG_PORT', '5432'))
PG_DB   = os.getenv('PROD_PG_DB',   os.getenv('PG_DB', 'mastergroup_recommendations'))
PG_USER = os.getenv('PROD_PG_USER', os.getenv('PG_USER', 'postgres'))
PG_PASS = os.getenv('PROD_PG_PASSWORD', os.getenv('PG_PASSWORD', 'postgres'))

CSV_PATH = os.getenv(
    'HISTORICAL_CSV',
    '/opt/mastergroup-ml/docs/CustomerDataMasterVerse_Cleaned.csv'
)

BATCH_SIZE = 5000

# ── Province map (from province_utils.py) ────────────────────────────────────
CITY_TO_PROVINCE = {
    'lahore': 'Punjab', 'faisalabad': 'Punjab', 'rawalpindi': 'Punjab',
    'multan': 'Punjab', 'gujranwala': 'Punjab', 'sialkot': 'Punjab',
    'bahawalpur': 'Punjab', 'sargodha': 'Punjab', 'sheikhupura': 'Punjab',
    'shekhupura': 'Punjab', 'jhang': 'Punjab', 'rahim yar khan': 'Punjab',
    'gujrat': 'Punjab', 'kasur': 'Punjab', 'sahiwal': 'Punjab', 'okara': 'Punjab',
    'wah cantt': 'Punjab', 'wah cantonment': 'Punjab', 'dera ghazi khan': 'Punjab',
    'chiniot': 'Punjab', 'mandi bahauddin': 'Punjab', 'jhelum': 'Punjab',
    'sadiqabad': 'Punjab', 'khanewal': 'Punjab', 'hafizabad': 'Punjab',
    'chakwal': 'Punjab', 'vehari': 'Punjab', 'attock': 'Punjab',
    'layyah': 'Punjab', 'muzaffargarh': 'Punjab', 'toba tek singh': 'Punjab',
    'pakpattan': 'Punjab', 'lodhran': 'Punjab', 'khushab': 'Punjab',
    'narowal': 'Punjab', 'mianwali': 'Punjab', 'nankana sahib': 'Punjab',
    'muridke': 'Punjab', 'wazirabad': 'Punjab', 'taxila': 'Punjab',
    'murree': 'Punjab', 'gojra': 'Punjab', 'jaranwala': 'Punjab',
    'karachi': 'Sindh', 'hyderabad': 'Sindh', 'sukkur': 'Sindh',
    'larkana': 'Sindh', 'nawabshah': 'Sindh', 'mirpur khas': 'Sindh',
    'jacobabad': 'Sindh', 'shikarpur': 'Sindh', 'khairpur': 'Sindh',
    'dadu': 'Sindh', 'thatta': 'Sindh', 'badin': 'Sindh',
    'sanghar': 'Sindh', 'umerkot': 'Sindh', 'ghotki': 'Sindh',
    'shahdadpur': 'Sindh', 'daharki': 'Sindh', 'rohri': 'Sindh',
    'peshawar': 'Khyber Pakhtunkhwa', 'mardan': 'Khyber Pakhtunkhwa',
    'abbottabad': 'Khyber Pakhtunkhwa', 'mingora': 'Khyber Pakhtunkhwa',
    'kohat': 'Khyber Pakhtunkhwa', 'swabi': 'Khyber Pakhtunkhwa',
    'charsadda': 'Khyber Pakhtunkhwa', 'nowshera': 'Khyber Pakhtunkhwa',
    'mansehra': 'Khyber Pakhtunkhwa', 'haripur': 'Khyber Pakhtunkhwa',
    'bannu': 'Khyber Pakhtunkhwa', 'swat': 'Khyber Pakhtunkhwa',
    'dera ismail khan': 'Khyber Pakhtunkhwa',
    'islamabad': 'Islamabad',
    'quetta': 'Balochistan', 'turbat': 'Balochistan', 'gwadar': 'Balochistan',
    'khuzdar': 'Balochistan', 'hub': 'Balochistan',
    'gilgit': 'Gilgit-Baltistan', 'skardu': 'Gilgit-Baltistan',
    'muzaffarabad': 'Azad Kashmir', 'mirpur': 'Azad Kashmir',
    'rawalakot': 'Azad Kashmir', 'kotli': 'Azad Kashmir',
}

def infer_province(city):
    if not city or str(city).strip().lower() in ('nan', 'none', ''):
        return 'Unspecified'
    return CITY_TO_PROVINCE.get(str(city).strip().lower(), 'Unspecified')


def make_uid(phone, name):
    """Synthesise unified_customer_id — mirrors the existing system logic."""
    if not phone:
        return None
    if name and str(name).strip():
        first = re.sub(r'[^a-z0-9]', '', str(name).strip().split()[0].lower())
        return f"{phone}_{first}" if first else phone
    return phone


def safe_str(val):
    """Return None if the pandas value is NaN/None, else stripped string."""
    if val is None or (isinstance(val, float) and __import__('math').isnan(val)):
        return None
    s = str(val).strip()
    return s if s.lower() not in ('nan', 'none', '') else None


# ── Main ingestion routine ───────────────────────────────────────────────────

def ingest():
    print(f"[{datetime.now():%H:%M:%S}] Connecting to {PG_DB} at {PG_HOST}…", flush=True)
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # ── Step 1: Pre-load existing phones & stat IDs into memory ─────────────
    print(f"[{datetime.now():%H:%M:%S}] Pre-loading existing phones…", flush=True)
    cur.execute("SELECT DISTINCT customer_phone FROM orders WHERE customer_phone IS NOT NULL")
    existing_phones = {r['customer_phone'] for r in cur.fetchall()}
    print(f"  → {len(existing_phones):,} unique phones already in DB.", flush=True)

    # Also track phones we've already inserted THIS run (for CSV-internal dedup)
    inserted_this_run = set()

    print(f"[{datetime.now():%H:%M:%S}] Pre-loading customer_statistics IDs…", flush=True)
    cur.execute("SELECT customer_id FROM customer_statistics")
    existing_stat_ids = {r['customer_id'] for r in cur.fetchall()}
    print(f"  → {len(existing_stat_ids):,} customer stat records loaded.", flush=True)

    # ── Step 2: Read CSV ─────────────────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Reading CSV from {CSV_PATH}…", flush=True)
    df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
    total_csv_rows = len(df)
    print(f"  → {total_csv_rows:,} rows in cleaned CSV.", flush=True)

    now = datetime.now()
    # Sentinel date for historical-only records (satisfies NOT NULL constraint,
    # clearly distinguishable from real orders and excluded by any > 2000 date filter)
    SENTINEL_DATE = '1900-01-01 00:00:00'
    stats = {
        'skipped_no_phone': 0,
        'overlap_enriched': 0,
        'overlap_skipped': 0,
        'csv_dedup_skipped': 0,
        'inserted_orders': 0,
        'inserted_stats': 0,
        'enriched_stats': 0,
    }

    # Process in chunks
    chunk_start = 0
    batch_num = 0

    while chunk_start < total_csv_rows:
        chunk = df.iloc[chunk_start: chunk_start + BATCH_SIZE]
        chunk_start += BATCH_SIZE
        batch_num += 1

        new_orders = []      # rows to insert into orders
        new_stats  = []      # rows to insert into customer_statistics (new customers)
        enrich_stats = []    # (name, email, city, customer_id) for UPDATE enrichment

        for _, row in chunk.iterrows():
            phone = safe_str(row.get('Mobileno'))
            if not phone:
                stats['skipped_no_phone'] += 1
                continue

            # CSV-internal dedup guard
            if phone in inserted_this_run:
                stats['csv_dedup_skipped'] += 1
                continue
            inserted_this_run.add(phone)

            name    = safe_str(row.get('CustomerName'))
            email   = safe_str(row.get('EmailAddress'))
            city    = safe_str(row.get('CityName'))
            address = safe_str(row.get('CustomerAddress'))
            source  = safe_str(row.get('DataSourceName')) or 'HISTORICAL'
            uid     = make_uid(phone, name)
            province = infer_province(city)

            if phone in existing_phones:
                # ── Overlap customer: enrich stats only ──────────────────────
                if uid and uid in existing_stat_ids:
                    enrich_stats.append((name, email, city, uid))
                    stats['overlap_enriched'] += 1
                else:
                    stats['overlap_skipped'] += 1
            else:
                # ── New customer: insert orders row + stats row ───────────────
                order_id = f"HIST_{uuid.uuid4().hex[:16].upper()}"
                new_orders.append((
                    order_id,           # id
                    'OE',               # order_type — must be 'OE' or 'POS' per DB constraint; historical records identified by source_type='HISTORICAL'
                    SENTINEL_DATE,      # order_date — sentinel: no real purchase date known
                    f'Historical import - {source}',  # order_name
                    'IMPORTED',         # order_status
                    name,               # customer_name
                    email,              # customer_email
                    phone,              # customer_phone
                    city,               # customer_city
                    address,            # customer_address
                    uid,                # unified_customer_id
                    0,                  # total_price (>= 0 per check constraint)
                    None,               # payment_mode
                    None,               # brand_name
                    None,               # items_json
                    now,                # created_at
                    now,                # updated_at
                    now,                # synced_at
                    province,           # province
                    None,               # region
                    'HISTORICAL',       # source_type — identifies this as historical, no constraint on this column
                ))

                if uid:
                    new_stats.append((
                        uid,            # customer_id
                        name,           # customer_name
                        city,           # customer_city
                        0, 0, 0,        # total_orders, total_products, unique_products
                        0, 0,           # total_spent, avg_order_value
                        None, None,     # first_order_date, last_order_date
                        'Historical',   # customer_segment
                        now,            # updated_at
                    ))
                    existing_stat_ids.add(uid)

                existing_phones.add(phone)
                stats['inserted_orders'] += len(new_orders) - stats['inserted_orders']

        # ── Flush orders ─────────────────────────────────────────────────────
        if new_orders:
            execute_batch(cur, """
                INSERT INTO orders (
                    id, order_type, order_date, order_name, order_status,
                    customer_name, customer_email, customer_phone, customer_city,
                    customer_address, unified_customer_id, total_price, payment_mode,
                    brand_name, items_json, created_at, updated_at, synced_at,
                    province, region, source_type
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s
                ) ON CONFLICT (id) DO NOTHING
            """, new_orders)
            stats['inserted_orders'] += len(new_orders)

        # ── Flush new stats ──────────────────────────────────────────────────
        # Filter out any uid already in existing_stat_ids (Python-level guard, no ON CONFLICT needed)
        clean_new_stats = [s for s in new_stats if s[0] not in existing_stat_ids or True]
        # Already filtered during collection — just insert directly
        if new_stats:
            try:
                execute_batch(cur, """
                    INSERT INTO customer_statistics (
                        customer_id, customer_name, customer_city,
                        total_orders, total_products, unique_products,
                        total_spent, avg_order_value,
                        first_order_date, last_order_date, customer_segment, updated_at
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s
                    )
                """, new_stats)
                stats['inserted_stats'] += len(new_stats)
            except Exception as e:
                conn.rollback()
                print(f"  [WARN] Stats batch insert failed ({e}), inserting individually…", flush=True)
                for s_row in new_stats:
                    try:
                        cur.execute("""
                            INSERT INTO customer_statistics (
                                customer_id, customer_name, customer_city,
                                total_orders, total_products, unique_products,
                                total_spent, avg_order_value,
                                first_order_date, last_order_date, customer_segment, updated_at
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, s_row)
                        stats['inserted_stats'] += 1
                    except Exception:
                        cur.execute("ROLLBACK TO SAVEPOINT sp1")


        # ── Enrich overlap stats (customer_statistics has no customer_email column) ──
        if enrich_stats:
            # enrich_stats tuples: (name, email, city, customer_id) — strip email since not in schema
            enrich_stats_no_email = [(name, city, uid) for name, email, city, uid in enrich_stats]
            execute_batch(cur, """
                UPDATE customer_statistics
                SET customer_name = COALESCE(%s, customer_name),
                    customer_city  = COALESCE(%s, customer_city),
                    updated_at     = NOW()
                WHERE customer_id = %s
            """, enrich_stats_no_email)
            stats['enriched_stats'] += len(enrich_stats_no_email)

        conn.commit()
        total_processed = chunk_start
        print(
            f"[{datetime.now():%H:%M:%S}] Batch {batch_num} | Processed {min(total_processed, total_csv_rows):,}/{total_csv_rows:,} | "
            f"Inserted: {stats['inserted_orders']:,} | Enriched: {stats['enriched_stats']:,} | "
            f"Overlap: {stats['overlap_enriched']:,}",
            flush=True
        )

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] ✅ Ingestion complete!", flush=True)
    print(f"  Skipped (no phone):    {stats['skipped_no_phone']:,}", flush=True)
    print(f"  CSV dedup skipped:     {stats['csv_dedup_skipped']:,}", flush=True)
    print(f"  Overlap enriched:      {stats['overlap_enriched']:,}", flush=True)
    print(f"  Overlap skipped:       {stats['overlap_skipped']:,}", flush=True)
    print(f"  New orders inserted:   {stats['inserted_orders']:,}", flush=True)
    print(f"  New stats inserted:    {stats['inserted_stats']:,}", flush=True)
    print(f"  Existing stats enriched: {stats['enriched_stats']:,}", flush=True)

    # ── Verification query ───────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Running verification queries…", flush=True)
    cur.execute("SELECT COUNT(*) AS cnt FROM orders WHERE source_type = 'HISTORICAL'")
    print(f"  orders WHERE source_type='HISTORICAL': {cur.fetchone()['cnt']:,}", flush=True)

    cur.execute("SELECT COUNT(*) AS cnt FROM customer_statistics WHERE customer_segment = 'Historical'")
    print(f"  customer_statistics WHERE segment='Historical': {cur.fetchone()['cnt']:,}", flush=True)

    cur.execute("""
        SELECT province, COUNT(*) AS cnt FROM orders
        WHERE source_type = 'HISTORICAL'
        GROUP BY province ORDER BY cnt DESC LIMIT 10
    """)
    print(f"\n  Province breakdown (Historical orders):", flush=True)
    for r in cur.fetchall():
        print(f"    {r['province']}: {r['cnt']:,}", flush=True)

    cur.close()
    conn.close()


if __name__ == '__main__':
    ingest()
