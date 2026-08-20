import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features import FEATURE_NAMES
from src.pipeline import build_feature_matrix, load_items_by_id, referenced_ids
from src.split import human_train_val_indices, valid_pairs_mask

LLM_BASE_WEIGHT = 1.0


def attach_ce_scores(feats, ce_scores_path, expected_rows, label):
    """Appends cross-encoder logits as a trailing feature column.

    The .npy must be aligned row-for-row with the matches file as read from
    disk — the length check is what catches a stale score file produced from
    a different or filtered matches file, which would otherwise silently
    misalign every score by an unknown offset.
    """
    if not ce_scores_path:
        return feats
    scores = np.load(ce_scores_path)
    if len(scores) != expected_rows:
        raise ValueError(
            f"{label} CE scores length {len(scores)} != {expected_rows} rows in the "
            f"matches file; regenerate them with src.ce_score against this exact file"
        )
    return np.hstack([feats, scores.reshape(-1, 1).astype(np.float32)])


def build_source(match_path, items_by_id, weight_fn, n_jobs, drop_targets=None,
                 exclude_items=None, ce_scores_path=None):
    match_df = pd.read_parquet(match_path)
    feats = build_feature_matrix(match_df, items_by_id, n_jobs=n_jobs)
    feats = attach_ce_scores(feats, ce_scores_path, len(match_df), "llm")
    target = match_df["target"].values

    if exclude_items:
        # Items already spent on the human validation split must not come
        # back in through an LLM-labeled row reusing the same item — that
        # would let the model see a near-duplicate of a held-out pair
        # during training. Applied after feature building so CE scores, which
        # are aligned to the unfiltered file, stay in step.
        touches_excluded = (
            match_df["id1"].isin(exclude_items) | match_df["id2"].isin(exclude_items)
        ).values
        feats = feats[~touches_excluded]
        target = target[~touches_excluded]

    # An all-NaN row across the *string* features means one of the pair's items
    # wasn't in items_by_id — no signal to train on (unlike at inference, we're
    # free to drop these). Restricted to those columns so the check doesn't
    # change meaning depending on whether a CE score column is attached.
    valid = ~np.isnan(feats[:, : len(FEATURE_NAMES)]).all(axis=1)
    if drop_targets is not None:
        valid &= ~np.isin(target, drop_targets)

    feats = feats[valid]
    target = target[valid]
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
    parser.add_argument("--human_weights", default="1,3,5,10",
                         help="comma-separated HUMAN_WEIGHT candidates to sweep; "
                              "the best one (by held-out human ROC-AUC) is kept")
    parser.add_argument("--ce_scores_path", default=None,
                         help="optional .npy of cross-encoder logits aligned to "
                              "matches_path, added as an extra feature column")
    parser.add_argument("--ce_scores_llm_path", default=None,
                         help="same for matches_llm_path")
    args = parser.parse_args()

    feature_names = list(FEATURE_NAMES)
    if args.ce_scores_path:
        feature_names.append("ce_score")

    # Both sources are vstacked into one training matrix, so either both carry
    # a CE column or neither does — otherwise the widths differ and the stack
    # fails (or worse, would silently misalign if the widths happened to match).
    if not args.skip_llm and bool(args.ce_scores_path) != bool(args.ce_scores_llm_path):
        parser.error(
            "--ce_scores_path and --ce_scores_llm_path must be given together "
            "(or use --skip_llm to train on human labels only)"
        )

    print("[1/6] Loading items (human subset)...")
    items_human = load_items_by_id(args.items_human_path)

    print("[2/6] Building features for human-labeled matches...")
    match_h_df = pd.read_parquet(args.matches_path)
    feats_h_all = build_feature_matrix(match_h_df, items_human, n_jobs=args.n_jobs)
    feats_h_all = attach_ce_scores(feats_h_all, args.ce_scores_path, len(match_h_df), "human")
    valid_h = valid_pairs_mask(match_h_df, items_human)
    match_h_df = match_h_df[valid_h].reset_index(drop=True)
    feats_h_all = feats_h_all[valid_h]
    y_h_all = (match_h_df["target"].values >= 0.5).astype(np.int32)
    print(f"  -> {len(y_h_all)} pairs, positive rate {y_h_all.mean():.3f}")
    del items_human

    print("[3/6] Holding out a validation split from human labels ONLY, "
          "grouped by connected item clusters...")
    train_idx, val_idx, groups_h = human_train_val_indices(
        match_h_df["id1"].values, match_h_df["id2"].values
    )
    print(f"  -> {len(np.unique(groups_h))} connected components among {len(groups_h)} human pairs")

    h_train_feats, h_val_feats = feats_h_all[train_idx], feats_h_all[val_idx]
    y_h_train, y_h_val = y_h_all[train_idx], y_h_all[val_idx]
    print(f"  -> human train={len(y_h_train)}, human val={len(y_h_val)} "
          f"(split by group, so not exactly 80/20)")

    val_items = set(match_h_df.iloc[val_idx]["id1"]) | set(match_h_df.iloc[val_idx]["id2"])
    del match_h_df, feats_h_all, y_h_all

    if args.skip_llm:
        feats_l = np.empty((0, len(feature_names)), dtype=np.float32)
        y_l = np.empty((0,), dtype=np.int32)
        w_l = np.empty((0,), dtype=np.float32)
    else:
        print("[4/6] Loading items (full set, for LLM-labeled matches)...")
        needed_ids = referenced_ids(args.matches_llm_path)
        print(f"  -> {len(needed_ids)} unique items referenced by matches_llm")
        items_full = load_items_by_id(args.items_full_path, required_ids=needed_ids)
        del needed_ids

        print("[5/6] Building features for LLM-labeled matches...")
        feats_l, y_l, w_l = build_source(
            args.matches_llm_path, items_full,
            # target==0.5 is maximally ambiguous — dropped via drop_targets
            # rather than merely down-weighted to 0, so it doesn't sit in the
            # matrix mislabeled as a hard positive (target >= 0.5).
            weight_fn=lambda t: LLM_BASE_WEIGHT * (2.0 * np.abs(t - 0.5)),
            n_jobs=args.n_jobs,
            drop_targets=[0.5],
            exclude_items=val_items,
            ce_scores_path=args.ce_scores_llm_path,
        )
        print(f"  -> {len(y_l)} pairs (target==0.5 and val-item overlap dropped), "
              f"positive rate {y_l.mean():.3f}")
        del items_full

    print("[6/6] Sweeping HUMAN_WEIGHT, scoring on held-out human labels only...")
    best = None
    for hw in (float(w) for w in args.human_weights.split(",")):
        w_train = np.concatenate([
            np.full(len(y_h_train), hw, dtype=np.float32),
            w_l,
        ])
        X_train = np.vstack([h_train_feats, feats_l])
        y_train = np.concatenate([y_h_train, y_l])

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

        val_pred = clf.predict_proba(h_val_feats)[:, 1]
        auc = roc_auc_score(y_h_val, val_pred)
        ap = average_precision_score(y_h_val, val_pred)
        print(f"  HUMAN_WEIGHT={hw:g}: human-val ROC-AUC={auc:.4f} PR-AUC={ap:.4f}")

        if best is None or auc > best[0]:
            best = (auc, hw, clf)

    best_auc, best_hw, best_clf = best
    print(f"Best HUMAN_WEIGHT={best_hw:g} (human-val ROC-AUC={best_auc:.4f})")

    print("Permutation importance (subsample of held-out human val set)...")
    sample_n = min(len(h_val_feats), 200_000)
    sample_idx = np.random.RandomState(1234).choice(len(h_val_feats), sample_n, replace=False)
    perm = permutation_importance(
        best_clf, h_val_feats[sample_idx], y_h_val[sample_idx],
        scoring="roc_auc", n_repeats=3, random_state=1234, n_jobs=args.n_jobs,
    )
    for name, imp in sorted(zip(feature_names, perm.importances_mean), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    joblib.dump(best_clf, args.output_model_path)
    print(f"Saved best model (HUMAN_WEIGHT={best_hw:g}) to {args.output_model_path}")


if __name__ == "__main__":
    main()
