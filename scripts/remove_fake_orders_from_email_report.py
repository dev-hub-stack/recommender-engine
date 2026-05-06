#!/usr/bin/env python3
"""Remove fake/test orders identified by the approved email audit report.

Safety model:
- Protected domains are excluded again in code and SQL.
- Matching orders, order_items, and customer_statistics are backed up first.
- Orders are then deleted, which removes their revenue from analytics.
- customer_statistics is rebuilt for affected customers from remaining orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_values


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_INPUT = REPORTS_DIR / "email_removal_candidates_from_latest.csv"
PROTECTED_DOMAINS = {"master.com.pk", "lums.edu.pk"}


def normalize_email(value: str | None) -> str:
    return (value or "").strip().lower()


def domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower() if "@" in email else ""


def read_candidates(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    emails: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        email_field = "email" if "email" in (reader.fieldnames or []) else (reader.fieldnames or [""])[0]
        for row in reader:
            email = normalize_email(row.get(email_field))
            if email and domain(email) not in PROTECTED_DOMAINS:
                emails.add(email)
    return sorted(emails)


def connection_params() -> dict[str, str | None]:
    load_dotenv(ROOT / ".env")
    return {
        "host": os.getenv("PROD_PG_HOST") or os.getenv("PG_HOST"),
        "port": os.getenv("PROD_PG_PORT") or os.getenv("PG_PORT", "5432"),
        "dbname": os.getenv("PROD_PG_DB") or os.getenv("PG_DB"),
        "user": os.getenv("PROD_PG_USER") or os.getenv("PG_USER"),
        "password": os.getenv("PROD_PG_PASSWORD") or os.getenv("PG_PASSWORD"),
    }


def fetch_one(cursor, sql: str, params: tuple = ()) -> dict:
    cursor.execute(sql, params)
    return dict(cursor.fetchone() or {})


def fetch_scalar(cursor, sql: str, params: tuple = ()):
    cursor.execute(sql, params)
    row = cursor.fetchone()
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0] if row else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Required for live deletion")
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing live deletion without --yes. Run dry-run first, then pass --yes.")

    candidates = read_candidates(args.input)
    if not candidates:
        raise SystemExit("No removable email candidates found after protected-domain filtering.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_orders_table = f"fake_order_cleanup_orders_backup_{run_id}"
    backup_items_table = f"fake_order_cleanup_order_items_backup_{run_id}"
    backup_stats_table = f"fake_order_cleanup_customer_stats_backup_{run_id}"
    report_base = f"fake_order_cleanup_execution_{run_id}"
    mode = "dry_run" if args.dry_run else "applied"

    conn = psycopg2.connect(**connection_params())
    try:
        conn.autocommit = False
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("CREATE TEMP TABLE tmp_fake_order_emails (email text PRIMARY KEY) ON COMMIT DROP")
            execute_values(
                cursor,
                "INSERT INTO tmp_fake_order_emails (email) VALUES %s ON CONFLICT DO NOTHING",
                [(email,) for email in candidates],
            )
            cursor.execute(
                "DELETE FROM tmp_fake_order_emails WHERE split_part(email, '@', 2) = ANY(%s)",
                (list(PROTECTED_DOMAINS),),
            )

            cursor.execute(
                """
                CREATE TEMP TABLE tmp_fake_orders AS
                SELECT o.id, o.unified_customer_id, LOWER(BTRIM(o.customer_email)) AS customer_email, o.total_price
                FROM orders o
                JOIN tmp_fake_order_emails e ON LOWER(BTRIM(o.customer_email)) = e.email
                WHERE split_part(LOWER(BTRIM(o.customer_email)), '@', 2) <> ALL(%s)
                """,
                (list(PROTECTED_DOMAINS),),
            )
            cursor.execute(
                """
                CREATE TEMP TABLE tmp_impacted_customers AS
                SELECT DISTINCT unified_customer_id AS customer_id
                FROM tmp_fake_orders
                WHERE unified_customer_id IS NOT NULL AND BTRIM(unified_customer_id) <> ''
                """
            )

            before = fetch_one(
                cursor,
                """
                SELECT
                    (SELECT COUNT(*) FROM orders) AS total_orders,
                    (SELECT COALESCE(SUM(total_price), 0) FROM orders) AS total_revenue,
                    (SELECT COUNT(*) FROM order_items) AS total_order_items,
                    (SELECT COUNT(*) FROM customer_statistics) AS customer_stat_rows
                """,
            )
            impact = fetch_one(
                cursor,
                """
                SELECT
                    COUNT(*) AS fake_order_rows,
                    COUNT(DISTINCT customer_email) AS fake_distinct_emails,
                    COUNT(DISTINCT unified_customer_id) FILTER (
                        WHERE unified_customer_id IS NOT NULL AND BTRIM(unified_customer_id) <> ''
                    ) AS impacted_customers,
                    COALESCE(SUM(total_price), 0) AS fake_revenue
                FROM tmp_fake_orders
                """,
            )
            item_impact = fetch_one(
                cursor,
                """
                SELECT
                    COUNT(*) AS fake_order_item_rows,
                    COALESCE(SUM(oi.total_price), 0) AS fake_order_item_revenue
                FROM order_items oi
                JOIN tmp_fake_orders f ON oi.order_id = f.id
                """,
            )
            protected = fetch_one(
                cursor,
                """
                SELECT
                    COUNT(*) AS protected_order_rows,
                    COUNT(DISTINCT LOWER(BTRIM(customer_email))) AS protected_distinct_emails
                FROM orders
                WHERE split_part(LOWER(BTRIM(customer_email)), '@', 2) = ANY(%s)
                """,
                (list(PROTECTED_DOMAINS),),
            )

            if not args.dry_run:
                cursor.execute(
                    f"""
                    CREATE TABLE {backup_orders_table} AS
                    SELECT o.*, NOW() AS cleanup_backed_up_at, %s AS cleanup_run_id
                    FROM orders o
                    JOIN tmp_fake_orders f ON f.id = o.id
                    """,
                    (run_id,),
                )
                cursor.execute(
                    f"""
                    CREATE TABLE {backup_items_table} AS
                    SELECT oi.*, NOW() AS cleanup_backed_up_at, %s AS cleanup_run_id
                    FROM order_items oi
                    JOIN tmp_fake_orders f ON f.id = oi.order_id
                    """,
                    (run_id,),
                )
                cursor.execute(
                    f"""
                    CREATE TABLE {backup_stats_table} AS
                    SELECT cs.*, NOW() AS cleanup_backed_up_at, %s AS cleanup_run_id
                    FROM customer_statistics cs
                    JOIN tmp_impacted_customers i ON i.customer_id = cs.customer_id
                    """,
                    (run_id,),
                )

                cursor.execute("DELETE FROM orders o USING tmp_fake_orders f WHERE o.id = f.id")
                deleted_orders = cursor.rowcount

                cursor.execute(
                    """
                    DELETE FROM customer_statistics cs
                    USING tmp_impacted_customers i
                    WHERE cs.customer_id = i.customer_id
                    """
                )
                deleted_stats = cursor.rowcount

                cursor.execute(
                    """
                    INSERT INTO customer_statistics (
                        customer_id, customer_name, customer_city, total_orders, total_products,
                        unique_products, total_spent, avg_order_value, first_order_date,
                        last_order_date, customer_segment, updated_at
                    )
                    SELECT
                        o.unified_customer_id AS customer_id,
                        MAX(o.customer_name) AS customer_name,
                        MAX(o.customer_city) AS customer_city,
                        COUNT(DISTINCT o.id)::int AS total_orders,
                        COALESCE(SUM(COALESCE(oi.quantity, 0)), 0)::int AS total_products,
                        COUNT(DISTINCT NULLIF(oi.product_id, ''))::int AS unique_products,
                        COALESCE(SUM(o.total_price), 0) AS total_spent,
                        COALESCE(AVG(o.total_price), 0) AS avg_order_value,
                        MIN(o.order_date) AS first_order_date,
                        MAX(o.order_date) AS last_order_date,
                        CASE
                            WHEN COUNT(DISTINCT o.id) >= 10 THEN 'VIP'
                            WHEN COUNT(DISTINCT o.id) >= 3 THEN 'Loyal'
                            WHEN COUNT(DISTINCT o.id) = 1 THEN 'New'
                            ELSE 'Regular'
                        END AS customer_segment,
                        NOW() AS updated_at
                    FROM orders o
                    JOIN tmp_impacted_customers i ON i.customer_id = o.unified_customer_id
                    LEFT JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.unified_customer_id IS NOT NULL AND BTRIM(o.unified_customer_id) <> ''
                    GROUP BY o.unified_customer_id
                    """
                )
                rebuilt_stats = cursor.rowcount
            else:
                deleted_orders = 0
                deleted_stats = 0
                rebuilt_stats = 0

            after = fetch_one(
                cursor,
                """
                SELECT
                    (SELECT COUNT(*) FROM orders) AS total_orders,
                    (SELECT COALESCE(SUM(total_price), 0) FROM orders) AS total_revenue,
                    (SELECT COUNT(*) FROM order_items) AS total_order_items,
                    (SELECT COUNT(*) FROM customer_statistics) AS customer_stat_rows
                """,
            )
            remaining = fetch_one(
                cursor,
                """
                SELECT
                    COUNT(*) AS remaining_fake_order_rows,
                    COALESCE(SUM(o.total_price), 0) AS remaining_fake_revenue
                FROM orders o
                JOIN tmp_fake_order_emails e ON LOWER(BTRIM(o.customer_email)) = e.email
                """
            )

            report = {
                "mode": mode,
                "run_id": run_id,
                "input_file": str(args.input),
                "protected_domains": sorted(PROTECTED_DOMAINS),
                "backup_tables": None if args.dry_run else {
                    "orders": backup_orders_table,
                    "order_items": backup_items_table,
                    "customer_statistics": backup_stats_table,
                },
                "impact_before_cleanup": {
                    "fake_order_rows": int(impact.get("fake_order_rows") or 0),
                    "fake_distinct_emails": int(impact.get("fake_distinct_emails") or 0),
                    "impacted_customers": int(impact.get("impacted_customers") or 0),
                    "fake_revenue_pkr": float(impact.get("fake_revenue") or 0),
                    "fake_order_item_rows": int(item_impact.get("fake_order_item_rows") or 0),
                    "fake_order_item_revenue_pkr": float(item_impact.get("fake_order_item_revenue") or 0),
                },
                "protected_left_unchanged": {
                    "order_rows": int(protected.get("protected_order_rows") or 0),
                    "distinct_emails": int(protected.get("protected_distinct_emails") or 0),
                },
                "changes_applied": {
                    "deleted_order_rows": int(deleted_orders),
                    "deleted_customer_stat_rows": int(deleted_stats),
                    "rebuilt_customer_stat_rows": int(rebuilt_stats),
                },
                "totals_before": {
                    "orders": int(before.get("total_orders") or 0),
                    "order_items": int(before.get("total_order_items") or 0),
                    "customer_stat_rows": int(before.get("customer_stat_rows") or 0),
                    "revenue_pkr": float(before.get("total_revenue") or 0),
                },
                "totals_after": {
                    "orders": int(after.get("total_orders") or 0),
                    "order_items": int(after.get("total_order_items") or 0),
                    "customer_stat_rows": int(after.get("customer_stat_rows") or 0),
                    "revenue_pkr": float(after.get("total_revenue") or 0),
                },
                "remaining_after_cleanup": {
                    "fake_order_rows": int(remaining.get("remaining_fake_order_rows") or 0),
                    "fake_revenue_pkr": float(remaining.get("remaining_fake_revenue") or 0),
                },
            }

            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()

        REPORTS_DIR.mkdir(exist_ok=True)
        json_payload = json.dumps(report, indent=2, sort_keys=True)
        json_path = REPORTS_DIR / f"{report_base}.json"
        md_path = REPORTS_DIR / f"{report_base}.md"
        latest_json = REPORTS_DIR / "fake_order_cleanup_execution_latest.json"
        latest_md = REPORTS_DIR / "fake_order_cleanup_execution_latest.md"
        json_path.write_text(json_payload + "\n", encoding="utf-8")
        latest_json.write_text(json_payload + "\n", encoding="utf-8")

        backup_lines = (
            "\n".join(f"- {name}: `{table}`" for name, table in report["backup_tables"].items())
            if report["backup_tables"]
            else "- not created (dry run)"
        )
        md = f"""# Fake Order Cleanup Execution Report

- Mode: `{mode}`
- Run ID: `{run_id}`
- Input file: `{args.input}`
- Protected domains left untouched: `{', '.join(sorted(PROTECTED_DOMAINS))}`

## Backups
{backup_lines}

## Removed From Analytics
- Fake order rows: `{report['impact_before_cleanup']['fake_order_rows']:,}`
- Fake order item rows: `{report['impact_before_cleanup']['fake_order_item_rows']:,}`
- Fake distinct emails: `{report['impact_before_cleanup']['fake_distinct_emails']:,}`
- Impacted customers: `{report['impact_before_cleanup']['impacted_customers']:,}`
- Fake revenue removed: `PKR {report['impact_before_cleanup']['fake_revenue_pkr']:,.2f}`

## Verification
- Orders before/after: `{report['totals_before']['orders']:,}` / `{report['totals_after']['orders']:,}`
- Order items before/after: `{report['totals_before']['order_items']:,}` / `{report['totals_after']['order_items']:,}`
- Revenue before/after: `PKR {report['totals_before']['revenue_pkr']:,.2f}` / `PKR {report['totals_after']['revenue_pkr']:,.2f}`
- Remaining fake order rows: `{report['remaining_after_cleanup']['fake_order_rows']:,}`
- Remaining fake revenue: `PKR {report['remaining_after_cleanup']['fake_revenue_pkr']:,.2f}`
- Protected order rows unchanged: `{report['protected_left_unchanged']['order_rows']:,}`
"""
        md_path.write_text(md, encoding="utf-8")
        latest_md.write_text(md, encoding="utf-8")
        print(json_payload)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
