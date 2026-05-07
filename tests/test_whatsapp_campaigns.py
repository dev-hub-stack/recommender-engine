import unittest
import sys
import types
from datetime import datetime
from unittest.mock import Mock

psycopg2_stub = types.ModuleType("psycopg2")
psycopg2_extras_stub = types.ModuleType("psycopg2.extras")
psycopg2_extras_stub.Json = lambda value: value
psycopg2_extras_stub.RealDictCursor = object
sys.modules.setdefault("psycopg2", psycopg2_stub)
sys.modules.setdefault("psycopg2.extras", psycopg2_extras_stub)

from src.services.whatsapp_campaigns import (
    MetaWhatsAppProvider,
    WhatsAppProviderConfig,
    WhatsAppProviderError,
    build_customer_message_context,
    build_smart_message_template,
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

    def test_meta_provider_blocks_numbers_outside_allowlist_in_test_mode(self):
        provider = MetaWhatsAppProvider(
            WhatsAppProviderConfig(
                provider_mode="test",
                access_token="token",
                phone_number_id="1074059625796986",
                default_template_name="hello_world",
                default_template_language="en_US",
                test_allowlist=["923214809481", "+923030644282"],
            )
        )

        with self.assertRaises(WhatsAppProviderError) as raised:
            provider.send_template_message(phone="923001111111")

        self.assertIn("not in WHATSAPP_TEST_ALLOWLIST", str(raised.exception))

    def test_meta_provider_posts_template_message_to_graph_api(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "messages": [{"id": "wamid.HBgM"}],
            "contacts": [{"wa_id": "923214809481"}],
        }
        session = Mock()
        session.post.return_value = response
        provider = MetaWhatsAppProvider(
            WhatsAppProviderConfig(
                provider_mode="test",
                access_token="token",
                phone_number_id="1074059625796986",
                default_template_name="hello_world",
                default_template_language="en_US",
                test_allowlist=["923214809481", "923030644282"],
            ),
            session=session,
        )

        result = provider.send_template_message(phone="0321 4809481")

        session.post.assert_called_once()
        url = session.post.call_args.args[0]
        headers = session.post.call_args.kwargs["headers"]
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(url, "https://graph.facebook.com/v20.0/1074059625796986/messages")
        self.assertEqual(headers["Authorization"], "Bearer token")
        self.assertEqual(payload["to"], "923214809481")
        self.assertEqual(payload["template"]["name"], "hello_world")
        self.assertEqual(payload["template"]["language"]["code"], "en_US")
        self.assertTrue(result["success"])
        self.assertEqual(result["provider_message_id"], "wamid.HBgM")

    def test_meta_provider_lists_approved_templates(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "data": [
                {
                    "name": "master_winback_v1",
                    "status": "APPROVED",
                    "language": "en_US",
                    "category": "MARKETING",
                    "components": [
                        {"type": "BODY", "text": "Hi {{1}}, based on {{2}}, try {{3}}."}
                    ],
                },
                {
                    "name": "draft_template",
                    "status": "PENDING",
                    "language": "en_US",
                    "category": "MARKETING",
                    "components": [{"type": "BODY", "text": "Pending"}],
                },
            ]
        }
        session = Mock()
        session.get.return_value = response
        provider = MetaWhatsAppProvider(
            WhatsAppProviderConfig(
                provider_mode="test",
                access_token="token",
                phone_number_id="1074059625796986",
                business_account_id="3149643718557386",
            ),
            session=session,
        )

        templates = provider.list_message_templates()

        session.get.assert_called_once()
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["name"], "master_winback_v1")
        self.assertEqual(templates[0]["body_parameter_count"], 3)

    def test_meta_provider_orders_numeric_template_variables(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"messages": [{"id": "wamid.HBgM"}]}
        session = Mock()
        session.post.return_value = response
        provider = MetaWhatsAppProvider(
            WhatsAppProviderConfig(
                provider_mode="test",
                access_token="token",
                phone_number_id="1074059625796986",
                default_template_name="master_winback_v1",
                test_allowlist=["923214809481"],
            ),
            session=session,
        )

        provider.send_template_message(
            phone="923214809481",
            variables={"2": "Celeste Mattress", "1": "Ali", "3": "Mattress Protector"},
        )

        payload = session.post.call_args.kwargs["json"]
        parameters = payload["template"]["components"][0]["parameters"]
        self.assertEqual([parameter["text"] for parameter in parameters], ["Ali", "Celeste Mattress", "Mattress Protector"])

    def test_meta_provider_creates_master_recommendation_template(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"id": "template-id", "status": "PENDING"}
        session = Mock()
        session.post.return_value = response
        provider = MetaWhatsAppProvider(
            WhatsAppProviderConfig(
                provider_mode="test",
                access_token="token",
                phone_number_id="1074059625796986",
                business_account_id="3149643718557386",
            ),
            session=session,
        )

        result = provider.create_message_template(
            name="master_recommendation_winback_v1",
            language="en_US",
            category="MARKETING",
            body_text="Hi {{1}}, based on {{2}}, try {{3}} with {{4}}: {{5}}",
            example_values=["Ali", "Celeste Mattress", "Mattress Protector", "MASTER10", "https://mastergroup.pk"],
        )

        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["name"], "master_recommendation_winback_v1")
        self.assertEqual(payload["category"], "MARKETING")
        self.assertEqual(payload["components"][0]["example"]["body_text"][0][2], "Mattress Protector")
        self.assertEqual(result["template"]["body_parameter_count"], 5)
        self.assertEqual(result["template"]["status"], "PENDING")

    def test_build_customer_message_context_uses_recent_purchase_and_recommendations(self):
        context = build_customer_message_context(
            {
                "customer_id": "cust-1",
                "customer_name": "Ali Khan",
                "customer_phone": "0303-0644282",
                "city": "Lahore",
                "segment": "At Risk",
                "recent_products": ["Celeste Mattress", "Memory Pillow"],
                "recommended_products": ["Mattress Protector", "Cooling Pillow"],
                "top_category": "Mattresses",
                "last_purchase_date": "2026-04-18",
                "total_orders": 3,
                "total_spend": 120000,
                "recency_days": 45,
            },
            discount_code="MASTER10",
            campaign_link="https://mastergroup.pk/campaign?utm_source=whatsapp",
        )

        self.assertEqual(context["customer_name"], "Ali Khan")
        self.assertEqual(context["phone"], "+923030644282")
        self.assertEqual(context["last_product"], "Celeste Mattress")
        self.assertEqual(context["recent_products"], ["Celeste Mattress", "Memory Pillow"])
        self.assertEqual(context["recommended_product_1"], "Mattress Protector")
        self.assertEqual(context["recommended_product_2"], "Cooling Pillow")
        self.assertEqual(context["top_category"], "Mattresses")
        self.assertEqual(context["segment"], "At Risk")

    def test_build_smart_message_template_is_segment_and_purchase_aware(self):
        template = build_smart_message_template("At Risk")

        self.assertIn("{{customer_name", template)
        self.assertIn("{{last_product", template)
        self.assertIn("{{recommended_product_1", template)
        self.assertIn("{{discount_code", template)


if __name__ == "__main__":
    unittest.main()
