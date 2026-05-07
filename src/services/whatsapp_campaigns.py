from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests
from psycopg2.extras import Json, RealDictCursor


CAMPAIGN_STATUSES = {
    "draft",
    "queued",
    "sending",
    "sent",
    "delivered",
    "read",
    "clicked",
    "failed",
    "opted_out",
}

DEFAULT_CAMPAIGN_FILTERS = {
    "segment": "Champions",
    "time_filter": "all",
    "order_source": "all",
    "categories": [],
    "statuses": [],
    "historical_channels": [],
    "exclude_invalid_phones": True,
    "dedupe_by_phone": True,
    "require_consent": False,
}


class WhatsAppProviderError(Exception):
    pass


@dataclass
class WhatsAppProviderConfig:
    provider_mode: str = "mock"
    access_token: Optional[str] = None
    phone_number_id: Optional[str] = None
    api_version: str = "v20.0"
    default_template_name: str = "hello_world"
    default_template_language: str = "en_US"
    test_allowlist: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "WhatsAppProviderConfig":
        allowlist = _list(os.getenv("WHATSAPP_TEST_ALLOWLIST"))
        return cls(
            provider_mode=(os.getenv("WHATSAPP_PROVIDER_MODE") or "mock").lower(),
            access_token=os.getenv("WHATSAPP_ACCESS_TOKEN"),
            phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID"),
            api_version=os.getenv("WHATSAPP_META_API_VERSION", "v20.0"),
            default_template_name=os.getenv("WHATSAPP_DEFAULT_TEMPLATE_NAME", "hello_world"),
            default_template_language=os.getenv("WHATSAPP_DEFAULT_TEMPLATE_LANGUAGE", "en_US"),
            test_allowlist=allowlist,
        )


class MetaWhatsAppProvider:
    provider_name = "meta"

    def __init__(self, config: WhatsAppProviderConfig, session: Optional[Any] = None):
        self.config = config
        self.session = session or requests.Session()

    def send_template_message(
        self,
        *,
        phone: Any,
        template_name: Optional[str] = None,
        template_language: Optional[str] = None,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        recipient = _whatsapp_api_phone(phone)
        if not recipient:
            raise WhatsAppProviderError("Invalid WhatsApp recipient phone")
        self._enforce_test_allowlist(recipient)

        if not self.config.access_token:
            raise WhatsAppProviderError("WHATSAPP_ACCESS_TOKEN is required")
        if not self.config.phone_number_id:
            raise WhatsAppProviderError("WHATSAPP_PHONE_NUMBER_ID is required")

        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "template",
            "template": {
                "name": template_name or self.config.default_template_name,
                "language": {"code": template_language or self.config.default_template_language},
            },
        }
        components = _template_components(variables or {})
        if components:
            payload["template"]["components"] = components

        response = self.session.post(
            f"https://graph.facebook.com/{self.config.api_version}/{self.config.phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {self.config.access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        data = response.json()
        if response.status_code >= 400:
            detail = data.get("error", {}).get("message") if isinstance(data, dict) else None
            raise WhatsAppProviderError(detail or f"Meta WhatsApp API failed with {response.status_code}")

        message_id = None
        if isinstance(data, dict) and data.get("messages"):
            message_id = data["messages"][0].get("id")
        return {
            "success": True,
            "provider": self.provider_name,
            "provider_mode": self.config.provider_mode,
            "provider_message_id": message_id,
            "recipient_phone": f"+{recipient}",
            "payload": data,
        }

    def _enforce_test_allowlist(self, recipient: str) -> None:
        if self.config.provider_mode != "test":
            return
        allowed = {_whatsapp_api_phone(phone) for phone in self.config.test_allowlist}
        allowed.discard(None)
        if recipient not in allowed:
            raise WhatsAppProviderError("Recipient is not in WHATSAPP_TEST_ALLOWLIST")


def get_whatsapp_provider_from_env() -> Optional[MetaWhatsAppProvider]:
    provider = (os.getenv("WHATSAPP_PROVIDER") or "mock").lower()
    if provider in {"", "mock"}:
        return None
    if provider != "meta":
        raise WhatsAppProviderError(f"Unsupported WhatsApp provider '{provider}'")
    return MetaWhatsAppProvider(WhatsAppProviderConfig.from_env())


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list(value: Any) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def normalize_phone(phone: Any) -> Optional[str]:
    """Normalize Pakistani mobile numbers for mock WhatsApp reachability checks."""
    if phone is None:
        return None
    digits = re.sub(r"[^\d]", "", str(phone).strip())
    if not digits:
        return None
    if len(digits) == 10 and digits.startswith("3"):
        return "+92" + digits
    if len(digits) == 11 and digits.startswith("03"):
        return "+92" + digits[1:]
    if len(digits) == 12 and digits.startswith("923"):
        return "+" + digits
    if len(digits) == 12 and digits.startswith("9203"):
        return "+923" + digits[4:]
    return None


def _whatsapp_api_phone(phone: Any) -> Optional[str]:
    normalized = normalize_phone(phone)
    if normalized:
        return normalized.lstrip("+")
    digits = re.sub(r"[^\d]", "", str(phone or "").strip())
    return digits or None


def _template_components(variables: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not variables:
        return []
    parameters = [
        {"type": "text", "text": str(value)}
        for _, value in sorted(variables.items())
        if value is not None
    ]
    if not parameters:
        return []
    return [{"type": "body", "parameters": parameters}]


def is_valid_phone(phone: Any) -> bool:
    return normalize_phone(phone) is not None


def normalize_campaign_filters(filters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = dict(DEFAULT_CAMPAIGN_FILTERS)
    if filters:
        normalized.update({k: v for k, v in filters.items() if v is not None})

    normalized["segment"] = _text(normalized.get("segment")) or DEFAULT_CAMPAIGN_FILTERS["segment"]
    normalized["time_filter"] = _text(normalized.get("time_filter")) or "all"
    normalized["order_source"] = (_text(normalized.get("order_source")) or "all").lower()
    normalized["categories"] = _list(normalized.get("categories"))
    normalized["statuses"] = _list(normalized.get("statuses"))
    normalized["historical_channels"] = _list(normalized.get("historical_channels"))
    normalized["exclude_invalid_phones"] = bool(normalized.get("exclude_invalid_phones", True))
    normalized["dedupe_by_phone"] = bool(normalized.get("dedupe_by_phone", True))
    normalized["require_consent"] = bool(normalized.get("require_consent", False))
    return normalized


def ensure_whatsapp_campaign_tables(conn) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS whatsapp_campaigns (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                message_template TEXT,
                filters JSONB NOT NULL DEFAULT '{}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                queued_at TIMESTAMP,
                sent_at TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS whatsapp_campaign_events (
                id SERIAL PRIMARY KEY,
                campaign_id INTEGER NOT NULL REFERENCES whatsapp_campaigns(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                recipient_phone TEXT,
                customer_id TEXT,
                provider TEXT NOT NULL DEFAULT 'mock',
                provider_message_id TEXT,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_whatsapp_campaign_events_campaign_time
            ON whatsapp_campaign_events(campaign_id, created_at DESC)
            """
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def _campaign_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    filters = row.get("filters") or {}
    metadata = row.get("metadata") or {}
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "message_template": row.get("message_template"),
        "filters": normalize_campaign_filters(filters),
        "metadata": metadata,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "queued_at": row.get("queued_at").isoformat() if row.get("queued_at") else None,
        "sent_at": row.get("sent_at").isoformat() if row.get("sent_at") else None,
    }


def create_campaign(conn, payload: Mapping[str, Any]) -> Dict[str, Any]:
    ensure_whatsapp_campaign_tables(conn)
    name = _text(payload.get("name"))
    if not name:
        raise ValueError("name is required")

    filters = normalize_campaign_filters(payload.get("filters") or {})
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            INSERT INTO whatsapp_campaigns (name, status, message_template, filters, metadata)
            VALUES (%s, 'draft', %s, %s::jsonb, %s::jsonb)
            RETURNING *
            """,
            (
                name,
                _text(payload.get("message_template")),
                Json(filters),
                Json(payload.get("metadata") or {}),
            ),
        )
        row = dict(cursor.fetchone())
        conn.commit()
        return _campaign_row(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def list_campaigns(conn, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    ensure_whatsapp_campaign_tables(conn)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        if status:
            cursor.execute(
                """
                SELECT *
                FROM whatsapp_campaigns
                WHERE status = %s
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (status, limit),
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM whatsapp_campaigns
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
        return [_campaign_row(dict(row)) for row in cursor.fetchall()]
    finally:
        cursor.close()


def get_campaign(conn, campaign_id: int) -> Optional[Dict[str, Any]]:
    ensure_whatsapp_campaign_tables(conn)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT * FROM whatsapp_campaigns WHERE id = %s", (campaign_id,))
        row = cursor.fetchone()
        return _campaign_row(dict(row)) if row else None
    finally:
        cursor.close()


def update_campaign(conn, campaign_id: int, payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    ensure_whatsapp_campaign_tables(conn)
    existing = get_campaign(conn, campaign_id)
    if not existing:
        return None
    if existing["status"] != "draft":
        raise ValueError("Only draft campaigns can be updated")

    name = _text(payload.get("name")) or existing["name"]
    message_template = (
        _text(payload.get("message_template"))
        if "message_template" in payload
        else existing.get("message_template")
    )
    filters = normalize_campaign_filters(payload.get("filters") or existing.get("filters") or {})
    metadata = payload.get("metadata") if "metadata" in payload else existing.get("metadata", {})

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            UPDATE whatsapp_campaigns
            SET name = %s,
                message_template = %s,
                filters = %s::jsonb,
                metadata = %s::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING *
            """,
            (name, message_template, Json(filters), Json(metadata or {}), campaign_id),
        )
        row = cursor.fetchone()
        conn.commit()
        return _campaign_row(dict(row)) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def record_campaign_event(
    conn,
    campaign_id: int,
    event_type: str,
    status: str,
    *,
    recipient_phone: Optional[str] = None,
    customer_id: Optional[str] = None,
    provider: str = "mock",
    provider_message_id: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_whatsapp_campaign_tables(conn)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            INSERT INTO whatsapp_campaign_events (
                campaign_id, event_type, status, recipient_phone, customer_id,
                provider, provider_message_id, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING *
            """,
            (
                campaign_id,
                event_type,
                status,
                normalize_phone(recipient_phone) or recipient_phone,
                customer_id,
                provider,
                provider_message_id or f"{provider}-{campaign_id}-{event_type}",
                Json(payload or {}),
            ),
        )
        row = dict(cursor.fetchone())
        conn.commit()
        return {
            "id": row["id"],
            "campaign_id": row["campaign_id"],
            "event_type": row["event_type"],
            "status": row["status"],
            "recipient_phone": row.get("recipient_phone"),
            "customer_id": row.get("customer_id"),
            "provider": row.get("provider"),
            "provider_message_id": row.get("provider_message_id"),
            "payload": row.get("payload") or {},
            "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def record_mock_event(
    conn,
    campaign_id: int,
    event_type: str,
    status: str,
    *,
    recipient_phone: Optional[str] = None,
    customer_id: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return record_campaign_event(
        conn,
        campaign_id,
        event_type,
        status,
        recipient_phone=recipient_phone,
        customer_id=customer_id,
        provider="mock",
        payload=payload,
    )


def mark_campaign_sent_mock(
    conn,
    campaign_id: int,
    *,
    audience_summary: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    ensure_whatsapp_campaign_tables(conn)
    campaign = get_campaign(conn, campaign_id)
    if not campaign:
        return None
    if campaign["status"] not in {"draft", "queued"}:
        raise ValueError(f"Campaign status {campaign['status']} cannot be sent")

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            UPDATE whatsapp_campaigns
            SET status = 'sent',
                queued_at = COALESCE(queued_at, CURRENT_TIMESTAMP),
                sent_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            RETURNING *
            """,
            (Json({"mock_send": True, "audience_summary": audience_summary or {}}), campaign_id),
        )
        row = dict(cursor.fetchone())
        conn.commit()
        return _campaign_row(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
