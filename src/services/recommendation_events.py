from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from psycopg2.extras import Json, RealDictCursor


VALID_EVENT_TYPES = {"impression", "click", "add_to_cart", "purchase"}
ATTRIBUTION_EVENT_TYPES = ("add_to_cart", "click", "impression")


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _decimal(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def normalize_event_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    event_type = _text(payload.get("event_type"))
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Unsupported event_type: {event_type}")

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw": metadata}

    for key, value in (
        ("page_url", payload.get("page_url")),
        ("cart_item_ids", payload.get("cart_item_ids")),
        ("recommended_product_name", payload.get("recommended_product_name")),
        ("recommendation_source", payload.get("recommendation_source")),
        ("attribution_source_widget", payload.get("attribution_source_widget")),
        ("attributed_from_click", payload.get("attributed_from_click")),
        ("score", payload.get("score")),
        ("country", payload.get("country")),
        ("market", payload.get("market")),
        ("location_source", payload.get("location_source")),
        ("location_confidence", payload.get("location_confidence")),
    ):
        if value not in (None, "", []):
            metadata[key] = value

    position = payload.get("position") or payload.get("recommended_position")
    if position in (None, ""):
        normalized_position = None
    else:
        normalized_position = int(position)

    seed_shopify_product_id = payload.get("seed_shopify_product_id") or payload.get("current_product")
    recommended_shopify_product_id = payload.get("recommended_shopify_product_id") or payload.get("shopify_product_id")

    return {
        "event_type": event_type,
        "occurred_at": _text(payload.get("occurred_at")),
        "mg_session_id": _text(payload.get("mg_session_id")),
        "rec_request_id": _text(payload.get("rec_request_id")),
        "storefront": _text(payload.get("storefront") or payload.get("store") or payload.get("store_domain")),
        "page_type": _text(payload.get("page_type")),
        "source_widget": _text(payload.get("source_widget")),
        "variant": _text(payload.get("variant") or payload.get("ab_variant")),
        "algorithm": _text(payload.get("algorithm")),
        "recommendation_type": _text(payload.get("recommendation_type") or payload.get("recommendation_source")),
        "seed_product_id": _text(payload.get("seed_product_id") or payload.get("seed_item_id")),
        "seed_shopify_product_id": _text(seed_shopify_product_id),
        "recommended_product_id": _text(
            payload.get("recommended_product_id")
            or payload.get("recommended_item_id")
            or payload.get("item_id")
        ),
        "recommended_shopify_product_id": _text(recommended_shopify_product_id),
        "shopify_handle": _text(payload.get("shopify_handle") or payload.get("recommended_product_handle")),
        "user_id": _text(payload.get("user_id") or payload.get("customer_id")),
        "customer_email": _text(payload.get("customer_email")),
        "customer_phone": _text(payload.get("customer_phone")),
        "city": _text(payload.get("city")),
        "province": _text(payload.get("province")),
        "position": normalized_position,
        "order_id": _text(payload.get("order_id")),
        "attributed_event_id": payload.get("attributed_event_id"),
        "revenue": _decimal(payload.get("revenue")),
        "metadata": metadata,
    }


def record_recommendation_event(conn, payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_event_payload(payload)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            INSERT INTO recommendation_events (
                event_type, occurred_at, mg_session_id, rec_request_id, storefront,
                page_type, source_widget, variant, algorithm, recommendation_type,
                seed_product_id, seed_shopify_product_id, recommended_product_id,
                recommended_shopify_product_id, shopify_handle, user_id,
                customer_email, customer_phone, city, province, position,
                order_id, attributed_event_id, revenue, metadata
            )
            VALUES (
                %(event_type)s,
                COALESCE(%(occurred_at)s::timestamp, CURRENT_TIMESTAMP),
                %(mg_session_id)s,
                %(rec_request_id)s,
                %(storefront)s,
                %(page_type)s,
                %(source_widget)s,
                %(variant)s,
                %(algorithm)s,
                %(recommendation_type)s,
                %(seed_product_id)s,
                %(seed_shopify_product_id)s::bigint,
                %(recommended_product_id)s,
                %(recommended_shopify_product_id)s::bigint,
                %(shopify_handle)s,
                %(user_id)s,
                %(customer_email)s,
                %(customer_phone)s,
                %(city)s,
                %(province)s,
                %(position)s,
                %(order_id)s,
                %(attributed_event_id)s,
                %(revenue)s,
                %(metadata)s::jsonb
            )
            RETURNING id, event_type, rec_request_id, mg_session_id, occurred_at
            """,
            {**normalized, "metadata": Json(normalized["metadata"])},
        )
        inserted = dict(cursor.fetchone())
        return inserted
    finally:
        cursor.close()


def record_recommendation_events(conn, payloads: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    inserted = []
    for payload in payloads:
        inserted.append(record_recommendation_event(conn, payload))
    return inserted


def _fetch_event_rows(cursor) -> List[Dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _run_event_query(conn, query: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(query, list(params))
        return _fetch_event_rows(cursor)
    finally:
        cursor.close()


def get_recommendation_events_by_session(
    conn,
    mg_session_id: str,
    *,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    session_id = _text(mg_session_id)
    if not session_id:
        raise ValueError("mg_session_id is required")

    return _run_event_query(
        conn,
        """
        SELECT *
        FROM recommendation_events
        WHERE mg_session_id = %s
        ORDER BY occurred_at DESC, id DESC
        LIMIT %s
        """,
        (session_id, limit),
    )


def get_recommendation_events_by_request(
    conn,
    rec_request_id: str,
    *,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    request_id = _text(rec_request_id)
    if not request_id:
        raise ValueError("rec_request_id is required")

    return _run_event_query(
        conn,
        """
        SELECT *
        FROM recommendation_events
        WHERE rec_request_id = %s
        ORDER BY occurred_at DESC, id DESC
        LIMIT %s
        """,
        (request_id, limit),
    )


def get_recommendation_events_by_order(
    conn,
    order_id: str,
    *,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    normalized_order_id = _text(order_id)
    if not normalized_order_id:
        raise ValueError("order_id is required")

    return _run_event_query(
        conn,
        """
        SELECT *
        FROM recommendation_events
        WHERE order_id = %s
        ORDER BY occurred_at DESC, id DESC
        LIMIT %s
        """,
        (normalized_order_id, limit),
    )


def find_attribution_candidates(
    conn,
    *,
    mg_session_id: Optional[str] = None,
    rec_request_id: Optional[str] = None,
    order_id: Optional[str] = None,
    user_id: Optional[str] = None,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
    recommended_shopify_product_id: Optional[str] = None,
    recommended_product_id: Optional[str] = None,
    attribution_window_days: int = 7,
    event_types: Sequence[str] = ATTRIBUTION_EVENT_TYPES,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    identity_filters = []
    identity_params: List[Any] = []

    normalized_session_id = _text(mg_session_id)
    normalized_request_id = _text(rec_request_id)
    normalized_order_id = _text(order_id)
    normalized_user_id = _text(user_id)
    normalized_email = _text(customer_email)
    normalized_phone = _text(customer_phone)

    if normalized_session_id:
        identity_filters.append("mg_session_id = %s")
        identity_params.append(normalized_session_id)
    if normalized_request_id:
        identity_filters.append("rec_request_id = %s")
        identity_params.append(normalized_request_id)
    if normalized_order_id:
        identity_filters.append("order_id = %s")
        identity_params.append(normalized_order_id)
    if normalized_user_id:
        identity_filters.append("user_id = %s")
        identity_params.append(normalized_user_id)
    if normalized_email:
        identity_filters.append("LOWER(customer_email) = LOWER(%s)")
        identity_params.append(normalized_email)
    if normalized_phone:
        identity_filters.append("customer_phone = %s")
        identity_params.append(normalized_phone)
    if not identity_filters:
        raise ValueError(
            "At least one attribution identifier is required: mg_session_id, rec_request_id, "
            "order_id, user_id, customer_email, or customer_phone"
        )

    normalized_event_types: List[str] = []
    for event_type in event_types:
        normalized_event_type = _text(event_type)
        if normalized_event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Unsupported attribution event_type: {normalized_event_type}")
        normalized_event_types.append(normalized_event_type)
    if not normalized_event_types:
        raise ValueError("At least one event_type is required")

    product_filters = []
    product_params: List[Any] = []
    if recommended_shopify_product_id:
        product_filters.append("recommended_shopify_product_id::text = %s")
        product_params.append(_text(recommended_shopify_product_id))
    if recommended_product_id:
        product_filters.append("recommended_product_id = %s")
        product_params.append(_text(recommended_product_id))

    query = f"""
        SELECT id, rec_request_id, mg_session_id, variant, algorithm, recommendation_type,
               recommended_product_id, recommended_shopify_product_id, shopify_handle,
               event_type, occurred_at, position, storefront, page_type, source_widget,
               user_id, customer_email, customer_phone, order_id, metadata
        FROM recommendation_events
        WHERE event_type IN ({', '.join(['%s'] * len(normalized_event_types))})
          AND occurred_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
          AND ({' OR '.join(identity_filters)})
    """

    params: List[Any] = [*normalized_event_types, attribution_window_days, *identity_params]

    if product_filters:
        query += f"\n          AND ({' OR '.join(product_filters)})"
        params.extend(product_params)

    query += """
        ORDER BY
            CASE event_type
                WHEN 'add_to_cart' THEN 0
                WHEN 'click' THEN 1
                WHEN 'impression' THEN 2
                ELSE 3
            END,
            occurred_at DESC,
            id DESC
        LIMIT %s
    """
    params.append(limit)

    return _run_event_query(conn, query, params)


def extract_shopify_note_attributes(order_payload: Mapping[str, Any]) -> Dict[str, str]:
    raw = order_payload.get("note_attributes") or {}
    attrs: Dict[str, str] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            key_text = _text(key)
            value_text = _text(value)
            if key_text and value_text:
                attrs[key_text] = value_text
        return attrs

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = _text(entry.get("name") or entry.get("key"))
        value = _text(entry.get("value"))
        if key and value:
            attrs[key] = value

    return attrs


def _find_best_attribution_event(
    cursor,
    *,
    mg_session_id: Optional[str],
    user_id: Optional[str],
    customer_email: Optional[str],
    customer_phone: Optional[str],
    recommended_shopify_product_id: Optional[str],
    recommended_product_id: Optional[str],
    attribution_window_days: int,
):
    conn = getattr(cursor, "connection", None)
    if conn is None:
        raise ValueError("cursor.connection is required for attribution lookup")

    matches = find_attribution_candidates(
        conn,
        mg_session_id=mg_session_id,
        user_id=user_id,
        customer_email=customer_email,
        customer_phone=customer_phone,
        recommended_shopify_product_id=recommended_shopify_product_id,
        recommended_product_id=recommended_product_id,
        attribution_window_days=attribution_window_days,
        limit=1,
    )
    return matches[0] if matches else None


def attribute_shopify_purchase(
    conn,
    order_payload: Mapping[str, Any],
    *,
    order_id: str,
    user_id: Optional[str] = None,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
    attribution_window_days: int = 7,
) -> List[Dict[str, Any]]:
    note_attrs = extract_shopify_note_attributes(order_payload)
    mg_session_id = note_attrs.get("mg_session_id") or note_attrs.get("_mg_session_id")

    line_items = order_payload.get("line_items") or []
    if not line_items:
        return []

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    attributed_events: List[Dict[str, Any]] = []
    try:
        for item in line_items:
            recommended_shopify_product_id = _text(item.get("product_id"))
            recommended_product_id = _text(item.get("sku")) or _text(item.get("title"))
            if not recommended_shopify_product_id and not recommended_product_id:
                continue

            cursor.execute(
                """
                SELECT id
                FROM recommendation_events
                WHERE event_type = 'purchase'
                  AND order_id = %s
                  AND (
                    (%s IS NOT NULL AND recommended_shopify_product_id::text = %s)
                    OR (%s IS NOT NULL AND recommended_product_id = %s)
                  )
                LIMIT 1
                """,
                (
                    order_id,
                    recommended_shopify_product_id,
                    recommended_shopify_product_id,
                    recommended_product_id,
                    recommended_product_id,
                ),
            )
            if cursor.fetchone():
                continue

            candidate = _find_best_attribution_event(
                cursor,
                mg_session_id=mg_session_id,
                user_id=user_id,
                customer_email=customer_email,
                customer_phone=customer_phone,
                recommended_shopify_product_id=recommended_shopify_product_id,
                recommended_product_id=None,
                attribution_window_days=attribution_window_days,
            )

            if not candidate:
                continue

            quantity = int(item.get("quantity") or 1)
            unit_price = _decimal(item.get("price")) or Decimal("0")
            revenue = unit_price * quantity

            event = record_recommendation_event(
                conn,
                {
                    "event_type": "purchase",
                    "mg_session_id": candidate.get("mg_session_id") or mg_session_id,
                    "rec_request_id": candidate.get("rec_request_id"),
                    "storefront": note_attrs.get("storefront") or "shopify",
                    "page_type": "checkout",
                    "source_widget": candidate.get("source_widget") or note_attrs.get("source_widget") or "shopify_webhook",
                    "variant": candidate.get("variant") or note_attrs.get("ab_variant"),
                    "algorithm": candidate.get("algorithm") or note_attrs.get("algorithm"),
                    "recommendation_type": candidate.get("recommendation_type"),
                    "recommended_product_id": candidate.get("recommended_product_id"),
                    "recommended_shopify_product_id": recommended_shopify_product_id,
                    "shopify_handle": candidate.get("shopify_handle"),
                    "user_id": user_id,
                    "customer_email": customer_email,
                    "customer_phone": customer_phone,
                    "city": _text((order_payload.get("shipping_address") or {}).get("city")),
                    "province": _text((order_payload.get("shipping_address") or {}).get("province")),
                    "order_id": order_id,
                    "attributed_event_id": candidate.get("id"),
                    "revenue": str(revenue),
                    "metadata": {
                        "shopify_order_id": _text(order_payload.get("id")),
                        "shopify_line_item_id": _text(item.get("id")),
                        "quantity": quantity,
                        "line_item_title": _text(item.get("title")),
                    },
                },
            )
            event["attributed_from_event_type"] = candidate.get("event_type")
            attributed_events.append(event)

        return attributed_events
    finally:
        cursor.close()
