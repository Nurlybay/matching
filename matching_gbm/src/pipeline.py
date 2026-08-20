import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, pair_features, prepare_item


def connected_component_groups(id1, id2):
    """Connected-component id per pair, over the graph where items are nodes
    and pairs are edges. Pairs sharing an item — or transitively connected
    through a chain of shared items, as near-duplicate clusters commonly are
    in product matching — land in the same group, so a group-aware split
    (GroupShuffleSplit) can keep each cluster entirely on one side instead of
    leaking near-identical feature vectors across train/val."""
    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(id1, id2):
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return np.fromiter((find(a) for a in id1), dtype=np.int64, count=len(id1))


def referenced_ids(match_path):
    """IDs actually touched by a matches file — filtering to this before
    loading the item catalog avoids paying memory for items that never
    appear in a pair."""
    df = pd.read_parquet(match_path, columns=["id1", "id2"])
    return set(df["id1"].values) | set(df["id2"].values)


def load_items_by_id(items_path, required_ids=None):
    """Loads only the raw (name, attributes, category) tuple per item —
    NOT the parsed/tokenized form. Parsing 12M+ items upfront (frozensets,
    dicts from JSON) is what was blowing past the container's memory limit;
    prepare_item() is instead called lazily, per pair, in _features_chunk."""
    df = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    if required_ids is not None:
        df = df[df["id"].isin(required_ids)]
    items_by_id = {}
    for item_id, name, attributes, category in zip(
        df["id"].values, df["name"].values, df["attributes"].values, df["category"].values
    ):
        items_by_id[item_id] = (name, attributes, category)
    return items_by_id


def _features_chunk(id1_chunk, id2_chunk, items_by_id):
    n = len(id1_chunk)
    # NaN, not np.empty: rows for pairs with a missing item are left as NaN
    # rather than dropped, so every row of match_df always gets a row here —
    # callers (run.py in particular) must be able to align predictions back
    # to id1/id2 1:1. HistGradientBoostingClassifier handles NaN features
    # natively, both for fit() and predict().
    feats = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    prepared_cache = {}  # scoped to this chunk only — dropped when the chunk ends

    def get_prepared(item_id):
        prepared = prepared_cache.get(item_id)
        if prepared is None:
            raw = items_by_id.get(item_id)
            if raw is None:
                return None
            prepared = prepare_item(item_id, *raw)
            prepared_cache[item_id] = prepared
        return prepared

    for i in range(n):
        it1 = get_prepared(id1_chunk[i])
        it2 = get_prepared(id2_chunk[i])
        if it1 is None or it2 is None:
            continue
        feats[i] = pair_features(it1, it2)
    return feats


def build_feature_matrix(match_df, items_by_id, n_jobs=1, chunk_size=200_000):
    """Processes pairs in fixed-size chunks so the per-item prepared-object
    cache stays bounded regardless of dataset size. Returns a feature matrix
    with exactly len(match_df) rows, in the same order — rows for pairs
    with a missing item are NaN, never dropped (see _features_chunk).

    n_jobs is currently ignored: an earlier multiprocessing.Pool(fork) version
    of this looked like free parallelism, but CPython bumps every touched
    object's refcount — a write — so "shared" pages get copy-on-write
    duplicated across workers as soon as they're read, not just written.
    With a ~70GB items dict that silently reproduced the exact memory blowup
    forking was meant to avoid, without any visible error. Sequential,
    chunked processing is slower but bounded and predictable; parallelizing
    this for real needs the item data in true shared memory (e.g. Arrow IPC
    /multiprocessing.shared_memory), not plain Python objects.
    """
    id1 = match_df["id1"].values
    id2 = match_df["id2"].values
    n = len(match_df)

    feats_parts = [
        _features_chunk(id1[i : i + chunk_size], id2[i : i + chunk_size], items_by_id)
        for i in range(0, n, chunk_size)
    ]
    return np.concatenate(feats_parts, axis=0)
