"""Helpers for source-aware live order identities."""

from __future__ import annotations

LIVE_ORDER_SOURCES = {"OE", "POS"}


def normalize_live_source(source: str | None) -> str:
    """Normalize live order source names used by Master Group feeds."""
    return (source or "").strip().upper()


def extract_source_order_id(order_id: str | int | None) -> str:
    """Return the upstream order ID without any source prefix."""
    raw = str(order_id or "").strip()
    if not raw:
        return ""
    for source in LIVE_ORDER_SOURCES:
        prefix = f"{source}:"
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def build_canonical_order_id(source: str | None, order_id: str | int | None) -> str:
    """Build the source-aware canonical ID for live OE/POS orders."""
    normalized_source = normalize_live_source(source)
    raw_order_id = extract_source_order_id(order_id)
    if not raw_order_id:
        return ""
    if normalized_source in LIVE_ORDER_SOURCES:
        return f"{normalized_source}:{raw_order_id}"
    return raw_order_id
