# Legacy SQL Files

⚠️ **These files are deprecated and kept for reference only.**

All database schema changes are now managed through Alembic migrations in the `migrations/versions/` directory.

## Do Not Use

These SQL files were used before the migration system was set up:

| File | Now Replaced By |
|------|-----------------|
| `deploy_database.sql` | `001_initial_schema.py` |
| `create_correct_orders_table.sql` | `001_initial_schema.py` |
| `setup_recommendation_tables.sql` | `001_initial_schema.py` + `003_stored_procedures.py` |
| `optimize_product_pairs.sql` | `003_stored_procedures.py` |
| `seed_users.sql` | `004_seed_admin_user.py` |

## Proper Setup

Use the new migration system instead:

```bash
# Run all migrations
python scripts/setup_database.py

# Or manually with Alembic
alembic upgrade head
```

See `DATABASE_SETUP.md` in the project root for complete instructions.
