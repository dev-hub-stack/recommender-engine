import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_storefront_name(value: Any) -> str:
    text = _text(value).lower()
    if not text:
        return ""
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_storefront_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value).lower())


def _mapping_rank(row: Mapping[str, Any]) -> tuple:
    return (
        1 if _text(row.get("shopify_handle")) else 0,
        1 if _text(row.get("shopify_image_url")) else 0,
        float(row.get("match_confidence") or 0),
        -len(_text(row.get("shopify_title"))),
        -len(_text(row.get("mastergroup_product_name"))),
    )


def build_shopify_mapping_indexes(mapping_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    indexes: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_mastergroup_id": {},
        "by_shopify_product_id": {},
        "by_handle": {},
        "by_sku": {},
        "by_name": {},
    }

    def register(index_name: str, key: str, row: Mapping[str, Any]) -> None:
        if not key:
            return
        current = indexes[index_name].get(key)
        candidate = dict(row)
        if current is None or _mapping_rank(candidate) > _mapping_rank(current):
            indexes[index_name][key] = candidate

    for row in mapping_rows:
        row_dict = dict(row)
        register("by_mastergroup_id", _text(row_dict.get("mastergroup_product_id")), row_dict)
        register("by_shopify_product_id", _text(row_dict.get("shopify_product_id")), row_dict)
        register("by_handle", normalize_storefront_token(row_dict.get("shopify_handle")), row_dict)
        register("by_sku", normalize_storefront_token(row_dict.get("shopify_sku")), row_dict)
        register("by_name", normalize_storefront_name(row_dict.get("mastergroup_product_name")), row_dict)
        register("by_name", normalize_storefront_name(row_dict.get("shopify_title")), row_dict)

    return indexes


def resolve_shopify_mapping(item: Mapping[str, Any], mapping_indexes: Dict[str, Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    item_id = _text(item.get("item_id") or item.get("product_id") or item.get("shopify_product_id"))
    item_name = _text(item.get("item_name") or item.get("product_name") or item.get("name"))
    item_token = normalize_storefront_token(item_id)

    if item_id:
        mapping = mapping_indexes["by_mastergroup_id"].get(item_id)
        if mapping:
            return mapping
        mapping = mapping_indexes["by_shopify_product_id"].get(item_id)
        if mapping:
            return mapping

    if item_token:
        mapping = mapping_indexes["by_sku"].get(item_token)
        if mapping:
            return mapping
        mapping = mapping_indexes["by_handle"].get(item_token)
        if mapping:
            return mapping

    for name_key in (normalize_storefront_name(item_name), normalize_storefront_name(item_id)):
        if not name_key:
            continue
        mapping = mapping_indexes["by_name"].get(name_key)
        if mapping:
            return mapping

    return None


def _build_current_context(
    mapping_indexes: Dict[str, Dict[str, Dict[str, Any]]],
    current_product_id: Optional[str] = None,
    current_internal_product_id: Optional[str] = None,
    current_handle: Optional[str] = None,
    current_title: Optional[str] = None,
) -> tuple[set[str], set[str], set[str]]:
    current_ids: set[str] = set()
    current_handles: set[str] = set()
    current_names: set[str] = set()

    def add_mapping(mapping: Optional[Mapping[str, Any]]) -> None:
        if not mapping:
            return
        for value in (
            mapping.get("shopify_product_id"),
            mapping.get("mastergroup_product_id"),
            mapping.get("shopify_sku"),
        ):
            text = _text(value)
            if text:
                current_ids.add(text)
        handle_token = normalize_storefront_token(mapping.get("shopify_handle"))
        if handle_token:
            current_handles.add(handle_token)
        for value in (mapping.get("shopify_title"), mapping.get("mastergroup_product_name")):
            normalized_name = normalize_storefront_name(value)
            if normalized_name:
                current_names.add(normalized_name)

    if current_product_id:
        add_mapping(
            mapping_indexes["by_shopify_product_id"].get(_text(current_product_id))
            or mapping_indexes["by_mastergroup_id"].get(_text(current_product_id))
        )
        current_ids.add(_text(current_product_id))

    if current_internal_product_id:
        add_mapping(mapping_indexes["by_mastergroup_id"].get(_text(current_internal_product_id)))
        current_ids.add(_text(current_internal_product_id))

    handle_token = normalize_storefront_token(current_handle)
    if handle_token:
        current_handles.add(handle_token)

    normalized_title = normalize_storefront_name(current_title)
    if normalized_title:
        current_names.add(normalized_title)

    return current_ids, current_handles, current_names


def _item_rank(item: Mapping[str, Any]) -> tuple:
    return (
        1 if _text(item.get("shopify_handle")) else 0,
        1 if _text(item.get("image_url")) else 0,
        1 if _text(item.get("shopify_product_id")) else 0,
        float(item.get("score") or 0),
    )


def prepare_shopify_storefront_items(
    raw_items: Iterable[Mapping[str, Any]],
    mapping_rows: Iterable[Mapping[str, Any]],
    *,
    current_product_id: Optional[str] = None,
    current_internal_product_id: Optional[str] = None,
    current_handle: Optional[str] = None,
    current_title: Optional[str] = None,
    limit: Optional[int] = None,
    require_handle: bool = False,
    mapping_indexes: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    indexes = mapping_indexes or build_shopify_mapping_indexes(mapping_rows)
    current_ids, current_handles, current_names = _build_current_context(
        indexes,
        current_product_id=current_product_id,
        current_internal_product_id=current_internal_product_id,
        current_handle=current_handle,
        current_title=current_title,
    )

    ordered_keys: List[str] = []
    chosen_items: Dict[str, Dict[str, Any]] = {}

    for raw_item in raw_items:
        item = dict(raw_item)
        item_id = _text(item.get("item_id") or item.get("product_id"))
        item_name = _text(item.get("item_name") or item.get("product_name") or item.get("name"))

        if item_id and not item.get("item_id"):
            item["item_id"] = item_id
        if item_name and not item.get("item_name"):
            item["item_name"] = item_name

        mapping = resolve_shopify_mapping(item, indexes)
        if mapping:
            if _text(mapping.get("shopify_product_id")):
                item["shopify_product_id"] = _text(mapping.get("shopify_product_id"))
            if _text(mapping.get("shopify_handle")):
                item["shopify_handle"] = _text(mapping.get("shopify_handle"))
            if _text(mapping.get("shopify_image_url")):
                item["image_url"] = _text(mapping.get("shopify_image_url"))
            if _text(mapping.get("shopify_title")):
                item["item_name"] = _text(mapping.get("shopify_title"))

        item_id = _text(item.get("item_id") or item.get("product_id"))
        shopify_product_id = _text(item.get("shopify_product_id"))
        item_handle = normalize_storefront_token(item.get("shopify_handle"))
        normalized_name = normalize_storefront_name(item.get("item_name") or item.get("product_name"))

        if item_id and item_id in current_ids:
            continue
        if shopify_product_id and shopify_product_id in current_ids:
            continue
        if item_handle and item_handle in current_handles:
            continue
        if normalized_name and normalized_name in current_names:
            continue
        if require_handle and not _text(item.get("shopify_handle")):
            continue

        dedupe_key = (
            normalize_storefront_token(item.get("shopify_handle"))
            or _text(item.get("shopify_product_id"))
            or normalized_name
            or item_id
        )
        if not dedupe_key:
            continue

        if dedupe_key not in chosen_items:
            chosen_items[dedupe_key] = item
            ordered_keys.append(dedupe_key)
            continue

        if _item_rank(item) > _item_rank(chosen_items[dedupe_key]):
            chosen_items[dedupe_key] = item

    prepared = [chosen_items[key] for key in ordered_keys]
    return prepared[:limit] if limit is not None else prepared
