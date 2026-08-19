import multiprocessing as mp

import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, pair_features, prepare_item

# Populated via set_items_by_id() before forking worker processes, so children
# inherit it through copy-on-write instead of each one paying to pickle/unpickle
# a multi-million-entry dict. Only correct with the "fork" start method
# (default on Linux) — on macOS/Windows n_jobs>1 silently falls back to n_jobs=1.
_ITEMS_BY_ID = None


def load_items_by_id(items_path):
    df = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items_by_id = {}
    for item_id, name, attributes, category in zip(
        df["id"].values, df["name"].values, df["attributes"].values, df["category"].values
    ):
        items_by_id[item_id] = prepare_item(item_id, name, attributes, category)
    return items_by_id


def _features_chunk(id1_chunk, id2_chunk, items_by_id):
    n = len(id1_chunk)
    feats = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        it1 = items_by_id.get(id1_chunk[i])
        it2 = items_by_id.get(id2_chunk[i])
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
    with ctx.Pool(processes=n_jobs) as pool:
        results = pool.map(_features_chunk_global, chunks)

    _ITEMS_BY_ID = None
    feats = np.concatenate([r[0] for r in results], axis=0)
    keep = np.concatenate([r[1] for r in results], axis=0)
    return feats, keep
