import json
import math
import re
from difflib import SequenceMatcher

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")

NAN = math.nan

FEATURE_NAMES = [
    "name_seq_ratio",
    "name_token_jaccard",
    "name_len_ratio",
    "name_len_diff",
    "common_word_count",
    "category_match",
    "attr_key_jaccard",
    "attr_common_key_count",
    "attr_exact_match_ratio",
    "attr_value_jaccard",
    "attr_count_diff",
    "attrs1_empty",
    "attrs2_empty",
]


def parse_attributes(raw):
    if not raw:
        return {}
    try:
        attrs = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(attrs, dict):
        return {}
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        out[str(k).strip().lower()] = str(v).strip().lower()
    return out


def tokenize(text):
    if not text:
        return frozenset()
    return frozenset(_TOKEN_RE.findall(text.lower()))


def prepare_item(item_id, name, attributes, category):
    """Precompute derived fields once per item so pair_features stays O(1) allocation-light."""
    name = str(name) if name is not None else ""
    attrs = parse_attributes(attributes)
    return {
        "id": item_id,
        "name": name,
        "name_tokens": tokenize(name),
        "category": str(category) if category is not None else "",
        "attrs": attrs,
        "attr_value_tokens": tokenize(" ".join(attrs.values())),
    }


def item_text(item, max_attrs=12):
    """Flat text for a transformer encoder.

    Attributes keep their original JSON order, which in this catalog tends to
    lead with the identifying fields (артикул, бренд, партномер) — so the
    max_attrs cut keeps the discriminative ones and drops boilerplate tails
    like «примечание» or «цена за».
    """
    parts = [item["name"]]
    if item["category"]:
        parts.append(item["category"])
    attrs = item["attrs"]
    if attrs:
        kv = list(attrs.items())[:max_attrs]
        parts.append(" ".join(f"{k}: {v}" for k, v in kv))
    return " | ".join(parts)


def _seq_ratio(a, b):
    # Both empty: nothing to compare, not a match — NaN, not a score.
    # Exactly one empty: a real, informative answer (they differ maximally).
    if not a and not b:
        return NAN
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _jaccard(a, b):
    if not a and not b:
        return NAN
    union = a | b
    if not union:
        return NAN
    return len(a & b) / len(union)


def pair_features(item1, item2):
    n1, n2 = item1["name"], item2["name"]
    t1, t2 = item1["name_tokens"], item2["name_tokens"]

    len1, len2 = len(n1), len(n2)
    max_len = max(len1, len2, 1)

    a1, a2 = item1["attrs"], item2["attrs"]
    shared_keys = a1.keys() & a2.keys()
    union_keys = a1.keys() | a2.keys()
    exact_matches = sum(1 for k in shared_keys if a1[k] == a2[k])

    # union_keys is empty exactly when both items have zero attributes —
    # no signal, so NaN rather than a fabricated "perfect match".
    attr_key_jaccard = (len(shared_keys) / len(union_keys)) if union_keys else NAN
    # shared_keys is empty both when both items have zero attributes AND when
    # they have disjoint (non-overlapping) key sets — in either case there is
    # no comparable key to score an exact-match rate over, so NaN, not 0.0
    # (0.0 would read as "checked and none matched", which isn't true here).
    attr_exact_match_ratio = (exact_matches / len(shared_keys)) if shared_keys else NAN

    return (
        _seq_ratio(n1, n2),
        _jaccard(t1, t2),
        min(len1, len2) / max_len,
        float(abs(len1 - len2)),
        float(len(t1 & t2)),
        float(item1["category"] == item2["category"]),
        attr_key_jaccard,
        float(len(shared_keys)),
        attr_exact_match_ratio,
        _jaccard(item1["attr_value_tokens"], item2["attr_value_tokens"]),
        float(abs(len(a1) - len(a2))),
        float(len(a1) == 0),
        float(len(a2) == 0),
    )
