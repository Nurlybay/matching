import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.pipeline import connected_component_groups

# Both the GBM and the cross-encoder must be scored on exactly the same
# held-out pairs, or their numbers aren't comparable — and worse, a pair held
# out from one model but trained on by the other leaks when their outputs are
# combined. These constants are the single source of truth for that split.
SPLIT_SEED = 1234
VAL_FRACTION = 0.2


def valid_pairs_mask(match_df, items_by_id):
    """Pairs where both items are actually present in the catalog."""
    id1 = match_df["id1"].values
    id2 = match_df["id2"].values
    known = items_by_id.__contains__
    return np.fromiter(
        (known(a) and known(b) for a, b in zip(id1, id2)),
        dtype=bool,
        count=len(match_df),
    )


def human_train_val_indices(id1, id2):
    """Deterministic group-aware split over connected components.

    Pairs sharing an item — or transitively linked through a chain of shared
    items, as near-duplicate product clusters commonly are — must stay on the
    same side, otherwise validation scores a model on near-copies of its own
    training rows. See README §5.2.
    """
    groups = connected_component_groups(id1, id2)
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SPLIT_SEED)
    train_idx, val_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
    return train_idx, val_idx, groups
