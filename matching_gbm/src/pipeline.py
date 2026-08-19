import multiprocessing as mp

import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, pair_features, prepare_item

# Populated via set_items_by_id() before forking worker processes, so children
# inherit it through copy-on-write instead of each one paying to pickle/unpickle
# a multi-million-entry dict. Only correct with the "fork" start method
# (default on Linux) — on macOS/Windows n_jobs>1 silently falls back to n_jobs=1.
_ITEMS_BY_ID = None


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
    feats = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    keep = np.ones(n, dtype=bool)
    prepared_cache = {}  # scoped to this chunk only — freed when the chunk returns

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
            keep[i] = False
            continue
        feats[i] = pair_features(it1, it2)
    return feats, keep


def _features_chunk_global(args):
    id1_chunk, id2_chunk = args
    return _features_chunk(id1_chunk, id2_chunk, _ITEMS_BY_ID)


def build_feature_matrix(match_df, items_by_id, n_jobs=1, chunk_size=200_000):
    id1 = match_df["id1"].values
    id2 = match_df["id2"].values
    n = len(match_df)

    can_fork = mp.get_start_method(allow_none=True) in (None, "fork") and mp.get_all_start_methods().count("fork") > 0
    if n_jobs <= 1 or n <= chunk_size or not can_fork:
        return _features_chunk(id1, id2, items_by_id)

    global _ITEMS_BY_ID
    _ITEMS_BY_ID = items_by_id  # set before Pool() so fork() copies it via COW

    ctx = mp.get_context("fork")
    chunks = [
        (id1[i : i + chunk_size], id2[i : i + chunk_size])
        for i in range(0, n, chunk_size)
    ]
    # fork() on a parent with a large RSS (the item catalog) pays a real, roughly
    # linear-in-process-count cost to duplicate page tables; forking more workers
    # than there is work for just adds that cost for nothing.
    n_workers = min(n_jobs, len(chunks))
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_features_chunk_global, chunks)

    _ITEMS_BY_ID = None
    feats = np.concatenate([r[0] for r in results], axis=0)
    keep = np.concatenate([r[1] for r in results], axis=0)
    return feats, keep
