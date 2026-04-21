import unittest

from src.services.shopify_storefront import (
    build_shopify_mapping_indexes,
    prepare_shopify_storefront_items,
    resolve_shopify_mapping,
)


class ShopifyStorefrontTests(unittest.TestCase):
    def setUp(self):
        self.mapping_rows = [
            {
                "shopify_product_id": 1001,
                "shopify_title": "Contour Pillow",
                "shopify_sku": "Contour-Pillow",
                "shopify_handle": "contour-pillow",
                "shopify_image_url": "https://img/contour.jpg",
                "mastergroup_product_id": "161757",
                "mastergroup_product_name": "Contour Pillow",
                "match_confidence": 1.0,
            },
            {
                "shopify_product_id": 1002,
                "shopify_title": "MoltyOrtho Back Care Cushion",
                "shopify_sku": "Back-Care-Cushion",
                "shopify_handle": "molty-back-care",
                "shopify_image_url": "https://img/back-care.jpg",
                "mastergroup_product_id": "161756",
                "mastergroup_product_name": "MoltyOrtho Back Care Cushion",
                "match_confidence": 1.0,
            },
            {
                "shopify_product_id": 1003,
                "shopify_title": "Master Foam",
                "shopify_sku": "master-foam-sku",
                "shopify_handle": "master-foam",
                "shopify_image_url": "",
                "mastergroup_product_id": "163104",
                "mastergroup_product_name": "Master Foam",
                "match_confidence": 0.8,
            },
            {
                "shopify_product_id": 1004,
                "shopify_title": "Master Foam",
                "shopify_sku": "master-foam-sku-alt",
                "shopify_handle": "master-foam",
                "shopify_image_url": "https://img/master-foam.jpg",
                "mastergroup_product_id": "162983",
                "mastergroup_product_name": "Master Foam",
                "match_confidence": 0.9,
            },
        ]
        self.mapping_indexes = build_shopify_mapping_indexes(self.mapping_rows)

    def test_resolve_shopify_mapping_uses_product_id_and_sku(self):
        direct = resolve_shopify_mapping({"item_id": "161756"}, self.mapping_indexes)
        self.assertEqual(direct["shopify_handle"], "molty-back-care")

        sku_match = resolve_shopify_mapping({"item_id": "Back-Care-Cushion"}, self.mapping_indexes)
        self.assertEqual(sku_match["shopify_handle"], "molty-back-care")

    def test_prepare_shopify_storefront_items_filters_self_and_dedupes(self):
        raw_items = [
            {"item_id": "161757", "item_name": "Contour Pillow", "score": 1.0},
            {"item_id": "161756", "item_name": "MoltyOrtho Back Care Cushion", "score": 0.9},
            {"item_id": "Back-Care-Cushion", "item_name": "MoltyOrtho Back Care Cushion (Back-Care-Cushion)", "score": 0.8},
            {"item_id": "163104", "item_name": "Master Foam", "score": 0.7},
            {"item_id": "162983", "item_name": "Master Foam", "score": 0.6},
        ]

        prepared = prepare_shopify_storefront_items(
            raw_items,
            self.mapping_rows,
            current_product_id="1001",
            current_internal_product_id="161757",
            current_handle="contour-pillow",
            current_title="Contour Pillow",
            require_handle=True,
            mapping_indexes=self.mapping_indexes,
        )

        self.assertEqual([item["shopify_handle"] for item in prepared], ["molty-back-care", "master-foam"])
        self.assertEqual(prepared[0]["item_name"], "MoltyOrtho Back Care Cushion")
        self.assertEqual(prepared[1]["image_url"], "https://img/master-foam.jpg")

    def test_prepare_shopify_storefront_items_can_require_handle(self):
        raw_items = [
            {"item_id": "unknown-item", "item_name": "Unknown Product", "score": 1.0},
            {"item_id": "161756", "item_name": "MoltyOrtho Back Care Cushion", "score": 0.9},
        ]

        prepared = prepare_shopify_storefront_items(
            raw_items,
            self.mapping_rows,
            require_handle=True,
            mapping_indexes=self.mapping_indexes,
        )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["shopify_handle"], "molty-back-care")


if __name__ == "__main__":
    unittest.main()
