import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features import item_text, prepare_item
from src.model_io import load_bundle
from src.pipeline import build_full_features, load_items_by_id
from src.split import human_train_val_indices, valid_pairs_mask


def describe(items_by_id, item_id, width):
    raw = items_by_id.get(item_id)
    if raw is None:
        return "<missing>"
    return item_text(prepare_item(item_id, *raw))[:width]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_human_path", default="items_human.parquet")
    parser.add_argument("--matches_path", default="matches.parquet")
    parser.add_argument("--ce_scores_path", default=None)
    parser.add_argument("--model_path", default="gbm_model.joblib")
    parser.add_argument("--n_examples", type=int, default=15)
    parser.add_argument("--width", type=int, default=110)
    args = parser.parse_args()

    print("Loading and rebuilding the validation split...")
    items_by_id = load_items_by_id(args.items_human_path)
    match_df = pd.read_parquet(args.matches_path)

    clf, category_map, _ = load_bundle(args.model_path)
    feats = build_full_features(
        match_df, items_by_id,
        category_map=category_map,
        ce_scores=np.load(args.ce_scores_path) if args.ce_scores_path else None,
        n_jobs=1,
    )

    mask = valid_pairs_mask(match_df, items_by_id)
    match_df = match_df[mask].reset_index(drop=True)
    feats = feats[mask]

    _, val_idx, _ = human_train_val_indices(match_df["id1"].values, match_df["id2"].values)
    val = match_df.iloc[val_idx].reset_index(drop=True)
    y = (val["target"].values >= 0.5).astype(int)

    p = clf.predict_proba(feats[val_idx])[:, 1]

    print(f"\n=== Overall (n={len(y)}, positive rate {y.mean():.3f}) ===")
    print(f"ROC-AUC {roc_auc_score(y, p):.4f}   PR-AUC {average_precision_score(y, p):.4f}")

    print("\n=== By category ===")
    # Category is shared within a pair (candidates are generated inside a
    # category), so id1's category labels the pair.
    cat = np.array([
        (items_by_id[i][2] if i in items_by_id else "?") for i in val["id1"].values
    ])
    rows = []
    for c in np.unique(cat):
        sel = cat == c
        if sel.sum() < 200 or len(np.unique(y[sel])) < 2:
            continue
        rows.append((roc_auc_score(y[sel], p[sel]), average_precision_score(y[sel], p[sel]),
                     int(sel.sum()), float(y[sel].mean()), c))
    rows.sort()
    print(f"{'ROC-AUC':>8} {'PR-AUC':>8} {'n':>7} {'pos':>6}  category")
    for auc, ap, n, pos, c in rows:
        print(f"{auc:8.4f} {ap:8.4f} {n:7d} {pos:6.3f}  {c}")

    print(f"\n=== Worst false positives (target=0, highest predicted) ===")
    neg = np.flatnonzero(y == 0)
    for i in neg[np.argsort(-p[neg])][: args.n_examples]:
        print(f"\np={p[i]:.3f}")
        print(f"  A: {describe(items_by_id, val['id1'][i], args.width)}")
        print(f"  B: {describe(items_by_id, val['id2'][i], args.width)}")

    print(f"\n=== Worst false negatives (target=1, lowest predicted) ===")
    pos = np.flatnonzero(y == 1)
    for i in pos[np.argsort(p[pos])][: args.n_examples]:
        print(f"\np={p[i]:.3f}")
        print(f"  A: {describe(items_by_id, val['id1'][i], args.width)}")
        print(f"  B: {describe(items_by_id, val['id2'][i], args.width)}")


if __name__ == "__main__":
    main()
