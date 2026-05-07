import unittest
import sys
import types
from datetime import datetime

psycopg2_stub = types.ModuleType("psycopg2")
psycopg2_extras_stub = types.ModuleType("psycopg2.extras")
psycopg2_extras_stub.Json = lambda value: value
psycopg2_extras_stub.RealDictCursor = object
sys.modules.setdefault("psycopg2", psycopg2_stub)
sys.modules.setdefault("psycopg2.extras", psycopg2_extras_stub)

from src.services.whatsapp_campaigns import (
    create_campaign,
    is_valid_phone,
    normalize_campaign_filters,
    normalize_phone,
    record_mock_event,
)


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.closed = False
        self.fetchone_result = None
        self.fetchall_result = []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if "INSERT INTO whatsapp_campaigns" in query:
            self.fetchone_result = {
                "id": 12,
                "name": "May Winback",
                "status": "draft",
                "message_template": "Hello {{name}}",
                "filters": {
                    "segment": "At Risk",
                    "time_filter": "30days",
                    "order_source": "oe",
                },
                "metadata": {},
                "created_at": datetime(2026, 5, 7, 10, 0),
                "updated_at": datetime(2026, 5, 7, 10, 0),
                "queued_at": None,
                "sent_at": None,
            }
        elif "INSERT INTO whatsapp_campaign_events" in query:
            self.fetchone_result = {
                "id": 44,
                "campaign_id": params[0],
                "event_type": params[1],
                "status": params[2],
                "recipient_phone": params[3],
                "customer_id": params[4],
                "provider": "mock",
                "provider_message_id": params[5],
                "payload": {"mock_mode": True},
                "created_at": datetime(2026, 5, 7, 10, 5),
            }

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class WhatsAppCampaignServiceTests(unittest.TestCase):
    def test_normalize_phone_accepts_pakistani_mobile_formats(self):
        self.assertEqual(normalize_phone("0300-1234567"), "+923001234567")
        self.assertEqual(normalize_phone("3001234567"), "+923001234567")
        self.assertEqual(normalize_phone("923001234567"), "+923001234567")
        self.assertTrue(is_valid_phone("+92 300 1234567"))
        self.assertFalse(is_valid_phone("12345"))

    def test_normalize_campaign_filters_lists_and_defaults(self):
        filters = normalize_campaign_filters(
            {
                "segment": "At Risk",
                "categories": "Mattresses,Pillows",
                "statuses": ["Delivered Orders"],
                "historical_channels": None,
            }
        )

        self.assertEqual(filters["segment"], "At Risk")
        self.assertEqual(filters["time_filter"], "all")
        self.assertEqual(filters["categories"], ["Mattresses", "Pillows"])
        self.assertEqual(filters["statuses"], ["Delivered Orders"])
        self.assertTrue(filters["exclude_invalid_phones"])
        self.assertTrue(filters["dedupe_by_phone"])

    def test_create_campaign_lazily_creates_tables_and_inserts_draft(self):
        conn = FakeConnection()

        campaign = create_campaign(
            conn,
            {
                "name": "May Winback",
                "message_template": "Hello {{name}}",
                "filters": {"segment": "At Risk", "order_source": "oe"},
            },
        )

        queries = [query for query, _ in conn.cursor_obj.executed]
        self.assertIn("CREATE TABLE IF NOT EXISTS whatsapp_campaigns", queries[0])
        self.assertTrue(any("INSERT INTO whatsapp_campaigns" in query for query in queries))
        self.assertEqual(campaign["id"], 12)
        self.assertEqual(campaign["status"], "draft")
        self.assertEqual(conn.commits, 2)

    def test_record_mock_event_never_uses_real_provider(self):
        conn = FakeConnection()

        event = record_mock_event(
            conn,
            12,
            "test_send",
            "sent",
            recipient_phone="03001234567",
            customer_id="cust-1",
            payload={"mock_mode": True},
        )

        self.assertEqual(event["provider"], "mock")
        self.assertEqual(event["recipient_phone"], "+923001234567")
        self.assertEqual(event["event_type"], "test_send")


if __name__ == "__main__":
    unittest.main()
