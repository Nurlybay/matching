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


def load_item_texts(items_path, required_ids, max_attrs=12, chunk_rows=1_000_000):
    """id -> encoder text, without ever materializing the raw-field dict.

    load_items_by_id followed by build_pair_texts holds both the raw tuples and
    the growing text cache at once, and that peak is what caps how much LLM
    data can be pretrained on — the measured lever on quality (§6). Formatting
    each row as it is read, in row-group batches, keeps only the texts.
    """
    import pyarrow.parquet as pq

    from src.features import item_text, prepare_item

    texts = {}
    parquet_file = pq.ParquetFile(items_path)
    for batch in parquet_file.iter_batches(
        batch_size=chunk_rows, columns=["id", "name", "attributes", "category"]
    ):
        ids = batch.column("id").to_pylist()
        names = batch.column("name").to_pylist()
        attrs = batch.column("attributes").to_pylist()
        cats = batch.column("category").to_pylist()
        for item_id, name, attribute, category in zip(ids, names, attrs, cats):
            if item_id in required_ids:
                texts[item_id] = item_text(
                    prepare_item(item_id, name, attribute, category), max_attrs=max_attrs
                )
    return texts


def category_codes(match_df, items_by_id, category_map):
    """Integer code of the pair's category, as a float column with NaN for
    unknown.

    Candidates are always generated inside one category, which is why the
    *agreement* feature `category_match` is constant and worthless — but which
    category it is turns out to matter a lot: measured per-category ROC-AUC
    ranges from 0.62 (Красота и гигиена) to 0.89 (Аптека). Without this the
    model is solving twenty rather different problems while blind to which one
    it is facing.

    NaN rather than a sentinel integer for unseen categories: HistGradient-
    BoostingClassifier handles missing values natively, whereas an unseen
    integer code would fall outside every bin learned at fit time.
    """
    codes = np.full(len(match_df), np.nan, dtype=np.float32)
    for row, item_id in enumerate(match_df["id1"].values):
        raw = items_by_id.get(item_id)
        if raw is None:
            continue
        code = category_map.get(raw[2])
        if code is not None:
            codes[row] = code
    return codes.reshape(-1, 1)


def build_full_features(match_df, items_by_id, category_map=None, ce_scores=None,
                        n_jobs=1, chunk_size=200_000):
    """Assembles the complete matrix in a fixed column order:
    string features, then category code, then CE score.

    Both training and inference go through this one function — the column
    order has to match exactly between them, and duplicating the assembly in
    two places is how that silently drifts.
    """
    feats = build_feature_matrix(match_df, items_by_id, n_jobs=n_jobs, chunk_size=chunk_size)
    parts = [feats]

    if category_map is not None:
        parts.append(category_codes(match_df, items_by_id, category_map))

    if ce_scores is not None:
        if len(ce_scores) != len(match_df):
            raise ValueError(
                f"CE scores length {len(ce_scores)} != {len(match_df)} rows in the "
                f"matches file; regenerate them against this exact file"
            )
        parts.append(ce_scores.reshape(-1, 1).astype(np.float32))

    return np.hstack(parts) if len(parts) > 1 else parts[0]


def full_feature_names(use_category=False, use_ce=False):
    names = list(FEATURE_NAMES)
    if use_category:
        names.append("category_code")
    if use_ce:
        names.append("ce_score")
    return names
