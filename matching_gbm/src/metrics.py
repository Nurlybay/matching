import numpy as np
from sklearn.metrics import average_precision_score


def macro_pr_auc(y_true, y_score, categories, min_count=1):
    """The competition metric: PR-AUC computed per category, then averaged.

    Deliberately not the pooled PR-AUC. Pooling weights a category by its share
    of pairs, while this weights all twenty equally — so a weak category with
    few pairs counts exactly as much as a strong one, and improving the weakest
    is worth more than polishing the strongest.

    Categories where one class is absent are skipped: average_precision_score
    is undefined there, and inventing a value would move the mean for reasons
    unrelated to the model.
    """
    scores = {}
    for cat in np.unique(categories):
        sel = categories == cat
        if sel.sum() < min_count or len(np.unique(y_true[sel])) < 2:
            continue
        scores[cat] = average_precision_score(y_true[sel], y_score[sel])
    if not scores:
        return float("nan"), {}
    return float(np.mean(list(scores.values()))), scores


def pair_categories(ids, cat_by_id):
    """Category label per pair, from an id -> category mapping. Candidates are
    generated inside one category, so either item's category identifies the
    pair."""
    return np.array([cat_by_id.get(i, "?") for i in ids])
