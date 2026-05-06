# Fake Order Cleanup Execution Report

- Mode: `applied`
- Run ID: `20260507_001657`
- Input file: `/Users/clustox1/Documents/Master Group/recommendation-engine-service/reports/email_removal_candidates_from_latest.csv`
- Protected domains left untouched: `lums.edu.pk, master.com.pk`

## Backups
- orders: `fake_order_cleanup_orders_backup_20260507_001657`
- order_items: `fake_order_cleanup_order_items_backup_20260507_001657`
- customer_statistics: `fake_order_cleanup_customer_stats_backup_20260507_001657`

## Removed From Analytics
- Fake order rows: `2,234`
- Fake order item rows: `910`
- Fake distinct emails: `566`
- Impacted customers: `1,219`
- Fake revenue removed: `PKR 113,551,733.91`

## Verification
- Orders before/after: `569,540` / `567,306`
- Order items before/after: `305,826` / `304,916`
- Revenue before/after: `PKR 12,164,468,471.43` / `PKR 12,050,916,737.52`
- Remaining fake order rows: `0`
- Remaining fake revenue: `PKR 0.00`
- Protected order rows unchanged: `1,651`
