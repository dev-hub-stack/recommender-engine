import unittest

from src.services.recommendation_events import (
    extract_shopify_note_attributes,
    find_attribution_candidates,
    get_recommendation_events_by_order,
    get_recommendation_events_by_request,
    get_recommendation_events_by_session,
    normalize_event_payload,
    record_recommendation_event,
)


class FakeCursor:
    def __init__(self, fetchone_result=None, fetchall_result=None):
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.executed = []
        self.closed = False
        self.connection = None

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self, cursor_factory=None):
        self._cursor.connection = self
        return self._cursor

    def close(self):
        self.closed = True


class RecommendationEventsTests(unittest.TestCase):
    def test_normalize_event_payload_maps_and_cleans_fields(self):
        payload = normalize_event_payload(
            {
                "event_type": "click",
                "mg_session_id": "  session-123  ",
                "store_domain": "masterverse-project.myshopify.com",
                "current_product": "1001",
                "recommended_item_id": "sku-44",
                "recommended_product_handle": "contour-pillow",
                "shopify_product_id": "2002",
                "recommended_position": "3",
                "recommendation_source": "similar",
                "metadata": {"foo": "bar"},
            }
        )

        self.assertEqual(payload["mg_session_id"], "session-123")
        self.assertEqual(payload["storefront"], "masterverse-project.myshopify.com")
        self.assertEqual(payload["seed_shopify_product_id"], "1001")
        self.assertEqual(payload["recommended_product_id"], "sku-44")
        self.assertEqual(payload["recommended_shopify_product_id"], "2002")
        self.assertEqual(payload["shopify_handle"], "contour-pillow")
        self.assertEqual(payload["position"], 3)
        self.assertEqual(payload["recommendation_type"], "similar")

    def test_normalize_event_payload_rejects_invalid_event_type(self):
        with self.assertRaises(ValueError):
            normalize_event_payload({"event_type": "hover"})

    def test_extract_shopify_note_attributes_handles_list_and_dict(self):
        attrs = extract_shopify_note_attributes(
            {
                "note_attributes": [
                    {"name": "mg_session_id", "value": "abc"},
                    {"key": "ab_variant", "value": "treatment"},
                ]
            }
        )
        self.assertEqual(attrs["mg_session_id"], "abc")
        self.assertEqual(attrs["ab_variant"], "treatment")

        attrs_from_dict = extract_shopify_note_attributes(
            {"note_attributes": {"mg_session_id": "xyz", "foo": "bar"}}
        )
        self.assertEqual(attrs_from_dict["mg_session_id"], "xyz")

    def test_record_recommendation_event_uses_named_insert_payload(self):
        cursor = FakeCursor(
            fetchone_result={
                "id": 77,
                "event_type": "add_to_cart",
                "mg_session_id": "sess-1",
                "recommended_product_id": "item-7",
            }
        )
        connection = FakeConnection(cursor)

        row = record_recommendation_event(
            connection,
            {
                "event_type": "add_to_cart",
                "mg_session_id": "sess-1",
                "recommended_product_id": "item-7",
                "shopify_handle": "molty-foam",
            },
        )

        self.assertEqual(row["id"], 77)
        query, params = cursor.executed[0]
        self.assertIn("INSERT INTO recommendation_events", query)
        self.assertEqual(params["event_type"], "add_to_cart")
        self.assertEqual(params["mg_session_id"], "sess-1")
        self.assertEqual(params["recommended_product_id"], "item-7")

    def test_query_helpers_filter_by_session_request_and_order(self):
        cursor = FakeCursor(fetchall_result=[{"id": 1}, {"id": 2}])
        connection = FakeConnection(cursor)

        rows = get_recommendation_events_by_session(connection, "session-42", limit=25)
        self.assertEqual(len(rows), 2)
        query, params = cursor.executed[0]
        self.assertIn("WHERE mg_session_id = %s", query)
        self.assertEqual(params, ["session-42", 25])

        cursor.executed.clear()
        get_recommendation_events_by_request(connection, "req-99", limit=10)
        query, params = cursor.executed[0]
        self.assertIn("WHERE rec_request_id = %s", query)
        self.assertEqual(params, ["req-99", 10])

        cursor.executed.clear()
        get_recommendation_events_by_order(connection, "ORDER-7", limit=5)
        query, params = cursor.executed[0]
        self.assertIn("WHERE order_id = %s", query)
        self.assertEqual(params, ["ORDER-7", 5])

    def test_find_attribution_candidates_builds_identity_and_product_filters(self):
        cursor = FakeCursor(fetchall_result=[{"id": 9, "event_type": "click"}])
        connection = FakeConnection(cursor)

        rows = find_attribution_candidates(
            connection,
            mg_session_id="session-123",
            customer_email="Buyer@Example.com",
            recommended_shopify_product_id="1001",
            recommended_product_id="item-1",
            limit=15,
        )

        self.assertEqual(rows[0]["id"], 9)
        query, params = cursor.executed[0]
        self.assertIn("event_type IN (%s, %s, %s)", query)
        self.assertIn("(mg_session_id = %s OR LOWER(customer_email) = LOWER(%s))", query)
        self.assertIn("(recommended_shopify_product_id::text = %s OR recommended_product_id = %s)", query)
        self.assertEqual(params[0:3], ["add_to_cart", "click", "impression"])
        self.assertEqual(params[4:6], ["session-123", "Buyer@Example.com"])
        self.assertEqual(params[6:8], ["1001", "item-1"])
        self.assertEqual(params[-1], 15)

    def test_find_attribution_candidates_requires_identifier(self):
        with self.assertRaises(ValueError):
            find_attribution_candidates(FakeConnection(FakeCursor()))


if __name__ == "__main__":
    unittest.main()
