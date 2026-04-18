#!/usr/bin/env python3
"""
Migrate live OE/POS orders to source-aware canonical IDs.

This rewrites:
- orders.id        -> OE:<source_order_id> / POS:<source_order_id>
- order_items.order_id to match

Historical / Shopify style records are left untouched.
"""

from __future__ import annotations

import os

import psycopg2
from psycopg2 import sql


def get_connection():
    host = os.getenv("PG_HOST", "localhost")
    sslmode = os.getenv("PG_SSLMODE", "prefer" if host == "localhost" else "require")
    return psycopg2.connect(
        host=host,
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DB", "mastergroup_recommendations"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        sslmode=sslmode,
    )


def fetch_fk_constraints(cursor):
    cursor.execute(
        """
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'order_items'::regclass
          AND confrelid = 'orders'::regclass
          AND contype = 'f'
        """
    )
    return [row[0] for row in cursor.fetchall()]


def main():
    conn = get_connection()
    conn.autocommit = False

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            ALTER TABLE orders
            ADD COLUMN IF NOT EXISTS source_order_id TEXT
            """
        )
        cursor.execute(
            """
            UPDATE orders
            SET source_type = order_type
            WHERE source_type IS NULL
              AND order_type IN ('OE', 'POS')
            """
        )
        cursor.execute(
            """
            UPDATE orders
            SET source_order_id = CASE
                WHEN id LIKE 'OE:%' OR id LIKE 'POS:%' THEN split_part(id, ':', 2)
                ELSE id
            END
            WHERE source_type IN ('OE', 'POS')
              AND (source_order_id IS NULL OR source_order_id = '')
            """
        )

        cursor.execute(
            """
            CREATE TEMP TABLE order_id_map AS
            SELECT
                id AS old_id,
                source_type || ':' || source_order_id AS new_id
            FROM orders
            WHERE source_type IN ('OE', 'POS')
              AND source_order_id IS NOT NULL
              AND source_order_id <> ''
              AND id <> source_type || ':' || source_order_id
            """
        )
        cursor.execute("SELECT COUNT(*) FROM order_id_map")
        rows_to_rewrite = cursor.fetchone()[0]
        print(f"rows_to_rewrite={rows_to_rewrite}")

        if rows_to_rewrite:
            fk_constraints = fetch_fk_constraints(cursor)
            for constraint_name in fk_constraints:
                cursor.execute(
                    sql.SQL("ALTER TABLE order_items DROP CONSTRAINT {}").format(
                        sql.Identifier(constraint_name)
                    )
                )

            cursor.execute(
                """
                UPDATE order_items oi
                SET order_id = m.new_id
                FROM order_id_map m
                WHERE oi.order_id = m.old_id
                """
            )
            print(f"order_items_rewritten={cursor.rowcount}")

            cursor.execute(
                """
                UPDATE orders o
                SET id = m.new_id
                FROM order_id_map m
                WHERE o.id = m.old_id
                """
            )
            print(f"orders_rewritten={cursor.rowcount}")

            cursor.execute(
                """
                ALTER TABLE order_items
                ADD CONSTRAINT order_items_order_id_fkey
                FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
                """
            )

        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_source_type_source_order_id
            ON orders(source_type, source_order_id)
            WHERE source_type IN ('OE', 'POS') AND source_order_id IS NOT NULL
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_orders_source_order_id
            ON orders(source_order_id)
            WHERE source_order_id IS NOT NULL
            """
        )

        conn.commit()
        print("migration_complete=true")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
