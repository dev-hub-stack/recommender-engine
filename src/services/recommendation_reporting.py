from __future__ import annotations

from typing import Any, Dict, List, Optional

from psycopg2.extras import RealDictCursor


def compute_rate(numerator: float, denominator: int) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / denominator, 6)


def compute_lift(treatment: float, control: float) -> float:
    if not control:
        return 0.0
    return round((treatment - control) / control, 6)


def summarize_counts(row: Dict[str, Any]) -> Dict[str, Any]:
    impressions = int(row.get("impressions") or 0)
    clicks = int(row.get("clicks") or 0)
    add_to_carts = int(row.get("add_to_carts") or 0)
    purchases = int(row.get("purchases") or 0)
    revenue = float(row.get("attributed_revenue") or 0)
    sessions = int(row.get("sessions") or 0)
    requests = int(row.get("recommendation_requests") or 0)

    return {
        "impressions": impressions,
        "clicks": clicks,
        "add_to_carts": add_to_carts,
        "purchases": purchases,
        "recommendation_requests": requests,
        "sessions": sessions,
        "attributed_revenue": revenue,
        "ctr": compute_rate(clicks, impressions),
        "add_to_cart_rate": compute_rate(add_to_carts, impressions),
        "purchase_conversion_rate": compute_rate(purchases, impressions),
        "click_to_cart_rate": compute_rate(add_to_carts, clicks),
        "click_to_purchase_rate": compute_rate(purchases, clicks),
        "revenue_per_impression": compute_rate(revenue, impressions),
        "revenue_per_click": compute_rate(revenue, clicks),
    }


def _build_filters(
    *,
    days: int,
    source_widget: Optional[str],
    page_type: Optional[str],
    variant: Optional[str],
    algorithm: Optional[str],
    province: Optional[str],
    city: Optional[str],
    storefront: Optional[str],
):
    clauses = ["occurred_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')"]
    params: List[Any] = [days]

    for column, value in (
        ("source_widget", source_widget),
        ("page_type", page_type),
        ("variant", variant),
        ("algorithm", algorithm),
        ("province", province),
        ("city", city),
        ("storefront", storefront),
    ):
        if value:
            clauses.append(f"{column} = %s")
            params.append(value)

    return " AND ".join(clauses), params


def _fetch_breakdown(cursor, field: str, where_sql: str, params: List[Any]) -> List[Dict[str, Any]]:
    cursor.execute(
        f"""
        SELECT
            COALESCE({field}, 'unknown') AS label,
            COUNT(*) FILTER (WHERE event_type = 'impression') AS impressions,
            COUNT(*) FILTER (WHERE event_type = 'click') AS clicks,
            COUNT(*) FILTER (WHERE event_type = 'add_to_cart') AS add_to_carts,
            COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases,
            COALESCE(SUM(revenue) FILTER (WHERE event_type = 'purchase'), 0) AS attributed_revenue,
            COUNT(DISTINCT mg_session_id) FILTER (WHERE mg_session_id IS NOT NULL) AS sessions,
            COUNT(DISTINCT rec_request_id) FILTER (WHERE rec_request_id IS NOT NULL) AS recommendation_requests
        FROM recommendation_events
        WHERE {where_sql}
        GROUP BY COALESCE({field}, 'unknown')
        ORDER BY clicks DESC, impressions DESC
        """,
        params,
    )
    rows = cursor.fetchall()
    return [
        {
            "label": row["label"],
            **summarize_counts(row),
        }
        for row in rows
    ]


def _build_treatment_lift(by_variant: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    variants = {row["label"]: row for row in by_variant}
    control = variants.get("control")
    treatment = variants.get("treatment")
    if not control or not treatment:
        return None

    return {
        "control": control,
        "treatment": treatment,
        "ctr_lift": compute_lift(treatment["ctr"], control["ctr"]),
        "add_to_cart_rate_lift": compute_lift(
            treatment["add_to_cart_rate"], control["add_to_cart_rate"]
        ),
        "purchase_conversion_rate_lift": compute_lift(
            treatment["purchase_conversion_rate"], control["purchase_conversion_rate"]
        ),
        "revenue_per_impression_lift": compute_lift(
            treatment["revenue_per_impression"], control["revenue_per_impression"]
        ),
    }


def build_recommendation_reporting_summary(
    conn,
    *,
    days: int = 30,
    source_widget: Optional[str] = None,
    page_type: Optional[str] = None,
    variant: Optional[str] = None,
    algorithm: Optional[str] = None,
    province: Optional[str] = None,
    city: Optional[str] = None,
    storefront: Optional[str] = None,
) -> Dict[str, Any]:
    where_sql, params = _build_filters(
        days=days,
        source_widget=source_widget,
        page_type=page_type,
        variant=variant,
        algorithm=algorithm,
        province=province,
        city=city,
        storefront=storefront,
    )

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE event_type = 'impression') AS impressions,
                COUNT(*) FILTER (WHERE event_type = 'click') AS clicks,
                COUNT(*) FILTER (WHERE event_type = 'add_to_cart') AS add_to_carts,
                COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases,
                COALESCE(SUM(revenue) FILTER (WHERE event_type = 'purchase'), 0) AS attributed_revenue,
                COUNT(DISTINCT mg_session_id) FILTER (WHERE mg_session_id IS NOT NULL) AS sessions,
                COUNT(DISTINCT rec_request_id) FILTER (WHERE rec_request_id IS NOT NULL) AS recommendation_requests
            FROM recommendation_events
            WHERE {where_sql}
            """,
            params,
        )
        overall = summarize_counts(dict(cursor.fetchone()))

        by_variant = _fetch_breakdown(cursor, "variant", where_sql, params)

        return {
            "filters": {
                "days": days,
                "source_widget": source_widget,
                "page_type": page_type,
                "variant": variant,
                "algorithm": algorithm,
                "province": province,
                "city": city,
                "storefront": storefront,
            },
            "overall": overall,
            "by_variant": by_variant,
            "by_algorithm": _fetch_breakdown(cursor, "algorithm", where_sql, params),
            "by_widget": _fetch_breakdown(cursor, "source_widget", where_sql, params),
            "by_page_type": _fetch_breakdown(cursor, "page_type", where_sql, params),
            "by_storefront": _fetch_breakdown(cursor, "storefront", where_sql, params),
            "by_province": _fetch_breakdown(cursor, "province", where_sql, params),
            "by_city": _fetch_breakdown(cursor, "city", where_sql, params),
            "treatment_vs_control": _build_treatment_lift(by_variant),
        }
    finally:
        cursor.close()
