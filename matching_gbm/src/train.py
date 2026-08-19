import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES
from src.pipeline import build_feature_matrix, load_items_by_id, referenced_ids

HUMAN_WEIGHT = 3.0
LLM_BASE_WEIGHT = 1.0


def build_source(match_path, items_by_id, weight_fn, n_jobs):
    match_df = pd.read_parquet(match_path)
    feats, keep = build_feature_matrix(match_df, items_by_id, n_jobs=n_jobs)
    target = match_df["target"].values[keep]
    feats = feats[keep]
    y_hard = (target >= 0.5).astype(np.int32)
    weight = weight_fn(target)
    return feats, y_hard, weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_human_path", default="items_human.parquet",
                         help="items covering the human-labeled matches file")
    parser.add_argument("--items_full_path", default="items.parquet",
                         help="full items file, needed to cover matches_llm pairs")
    parser.add_argument("--matches_path", default="matches.parquet")
    parser.add_argument("--matches_llm_path", default="matches_llm.parquet")
    parser.add_argument("--output_model_path", default="gbm_model.joblib")
    parser.add_argument("--n_jobs", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--skip_llm", action="store_true",
                         help="train on human labels only")
    args = parser.parse_args()

    print("[1/5] Loading items (human subset)...")
    items_human = load_items_by_id(args.items_human_path)

    print("[2/5] Building features for human-labeled matches...")
    feats_h, y_h, w_h = build_source(
        args.matches_path, items_human,
        weight_fn=lambda t: np.full(len(t), HUMAN_WEIGHT, dtype=np.float32),
        n_jobs=args.n_jobs,
    )
    print(f"  -> {len(y_h)} pairs, positive rate {y_h.mean():.3f}")
    del items_human

    if args.skip_llm:
        X, y, w = feats_h, y_h, w_h
    else:
        print("[3/5] Loading items (full set, for LLM-labeled matches)...")
        needed_ids = referenced_ids(args.matches_llm_path)
        print(f"  -> {len(needed_ids)} unique items referenced by matches_llm")
        items_full = load_items_by_id(args.items_full_path, required_ids=needed_ids)
        del needed_ids

        print("[4/5] Building features for LLM-labeled matches...")
        feats_l, y_l, w_l = build_source(
            args.matches_llm_path, items_full,
            weight_fn=lambda t: LLM_BASE_WEIGHT * (2.0 * np.abs(t - 0.5)),
            n_jobs=args.n_jobs,
        )
        print(f"  -> {len(y_l)} pairs, positive rate {y_l.mean():.3f}")
        del items_full

        X = np.vstack([feats_h, feats_l])
        y = np.concatenate([y_h, y_l])
        w = np.concatenate([w_h, w_l])

    print("[5/5] Training HistGradientBoostingClassifier...")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w, test_size=0.1, random_state=1234, stratify=y
    )

    clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=1234,
    )
    clf.fit(X_train, y_train, sample_weight=w_train)

    val_pred = clf.predict_proba(X_val)[:, 1]
    print("val ROC-AUC:", roc_auc_score(y_val, val_pred))
    print("val PR-AUC:", average_precision_score(y_val, val_pred))

    print("Permutation importance (subsample of val set)...")
    sample_n = min(len(X_val), 200_000)
    sample_idx = np.random.RandomState(1234).choice(len(X_val), sample_n, replace=False)
    perm = permutation_importance(
        clf, X_val[sample_idx], y_val[sample_idx],
        scoring="roc_auc", n_repeats=3, random_state=1234, n_jobs=args.n_jobs,
    )
    for name, imp in sorted(zip(FEATURE_NAMES, perm.importances_mean), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    joblib.dump(clf, args.output_model_path)
    print(f"Saved model to {args.output_model_path}")


if __name__ == "__main__":
    main()
