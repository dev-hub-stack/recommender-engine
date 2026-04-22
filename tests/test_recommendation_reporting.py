import unittest

from src.services.recommendation_reporting import compute_lift, compute_rate, summarize_counts


class RecommendationReportingTests(unittest.TestCase):
    def test_compute_rate_handles_zero_denominator(self):
        self.assertEqual(compute_rate(1, 0), 0.0)

    def test_compute_lift_handles_zero_control(self):
        self.assertEqual(compute_lift(0.1, 0.0), 0.0)

    def test_summarize_counts_builds_core_metrics(self):
        summary = summarize_counts(
            {
                "impressions": 100,
                "clicks": 12,
                "add_to_carts": 5,
                "purchases": 2,
                "attributed_revenue": 1500,
                "sessions": 40,
                "recommendation_requests": 25,
            }
        )

        self.assertEqual(summary["impressions"], 100)
        self.assertEqual(summary["clicks"], 12)
        self.assertEqual(summary["ctr"], 0.12)
        self.assertEqual(summary["add_to_cart_rate"], 0.05)
        self.assertEqual(summary["purchase_conversion_rate"], 0.02)
        self.assertEqual(summary["click_to_cart_rate"], 0.416667)
        self.assertEqual(summary["click_to_purchase_rate"], 0.166667)
        self.assertEqual(summary["revenue_per_impression"], 15.0)
        self.assertEqual(summary["revenue_per_click"], 125.0)
        self.assertEqual(summary["attributed_revenue"], 1500.0)


if __name__ == "__main__":
    unittest.main()
