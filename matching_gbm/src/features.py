import json
import math
import re
from difflib import SequenceMatcher

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
# Numbers carry the distinguishing detail in this catalog far more often than
# their share of the text suggests: diopters (-2.00 vs -3.00), curtain sizes
# (300х220 vs 220), РЦ ranges (66-68 vs 68-70). Captured with decimals so
# "1.56" stays one number rather than splitting into 1 and 56.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Latin/Cyrillic spellings of the same brand coexist throughout the data
# ("asics" / "асикс"), and token-level features score those as zero overlap.
# Folding Cyrillic onto Latin makes them comparable — approximately, which is
# enough since the comparison downstream is fuzzy anyway.
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "c", "ш": "s", "щ": "s", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "u", "я": "a",
})

# Attribute keys that identify a specific manufactured article. A match on one
# of these is close to decisive, and it is exactly the signal a subword
# tokenizer destroys — which is why Автотовары, where these dominate, barely
# improved when the cross-encoder was added (0.6687 -> 0.6734 vs +0.128 for
# Красота и гигиена).
_ID_KEYS = (
    "артикул",
    "артикул производителя",
    "партномер (артикул производителя)",
    "oem-номер",
    "код товара",
    "модель",
)

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
    "name_translit_ratio",
    "number_jaccard",
    "number_count_diff",
    "id_attr_match",
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


def transliterate(text):
    return text.lower().translate(_TRANSLIT)


def numbers(text):
    """Numeric tokens, comma normalized to a dot so "1,56" and "1.56" agree."""
    return frozenset(m.group().replace(",", ".") for m in _NUMBER_RE.finditer(text))


def identifiers(attrs):
    """Non-empty values of the identifying attribute keys, normalized."""
    out = set()
    for key in _ID_KEYS:
        value = attrs.get(key)
        if value and value not in ("", "-", "нет"):
            # Separators are inconsistent across sellers ("8500892sx" vs
            # "8500892-sx"), so compare on alphanumerics only.
            out.add("".join(ch for ch in value if ch.isalnum()))
    return out


def prepare_item(item_id, name, attributes, category):
    """Precompute derived fields once per item so pair_features stays O(1) allocation-light."""
    name = str(name) if name is not None else ""
    attrs = parse_attributes(attributes)
    return {
        "id": item_id,
        "name": name,
        "name_tokens": tokenize(name),
        "name_translit": transliterate(name),
        "name_numbers": numbers(name),
        "category": str(category) if category is not None else "",
        "attrs": attrs,
        "attr_value_tokens": tokenize(" ".join(attrs.values())),
        "identifiers": identifiers(attrs),
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


def _id_match(ids1, ids2):
    """1.0 if the two items share an identifier, 0.0 if both have identifiers
    but none coincide, NaN if either side has none to compare.

    The NaN case is the majority — most listings carry no артикул — and
    collapsing it into 0.0 would tell the model "checked, they differ", which
    is a different and false claim.
    """
    if not ids1 or not ids2:
        return NAN
    return float(bool(ids1 & ids2))


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
        _seq_ratio(item1["name_translit"], item2["name_translit"]),
        _jaccard(item1["name_numbers"], item2["name_numbers"]),
        float(abs(len(item1["name_numbers"]) - len(item2["name_numbers"]))),
        _id_match(item1["identifiers"], item2["identifiers"]),
    )
