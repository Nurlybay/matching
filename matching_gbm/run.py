import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd

from src.pipeline import build_feature_matrix, load_items_by_id

MODEL_PATH = "gbm_model.joblib"
CE_MODEL_DIR = "ce_model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    parser.add_argument("--ce_batch_size", type=int, default=256)
    parser.add_argument("--ce_max_length", type=int, default=192)
    args = parser.parse_args()

    t0 = time.time()
    use_ce = os.path.isdir(CE_MODEL_DIR)

    print("[1/4] Loading items...")
    items_by_id = load_items_by_id(args.items_path)
    match_df = pd.read_parquet(args.matches_path)
    print(f"  -> {len(items_by_id)} items, {len(match_df)} pairs ({time.time() - t0:.1f}s)")

    print("[2/4] Building string features...")
    feats = build_feature_matrix(match_df, items_by_id, n_jobs=1)
    print(f"  -> {feats.shape} ({time.time() - t0:.1f}s)")

    if use_ce:
        # Imported lazily so a GBM-only submission never pays the torch import
        # or CUDA init, which matter against the 1-minute check-stage limit.
        from src.ce_score import score_match_df

        print("[3/4] Scoring pairs with the cross-encoder...")
        ce_scores = score_match_df(
            match_df, items_by_id, CE_MODEL_DIR,
            batch_size=args.ce_batch_size, max_length=args.ce_max_length,
        )
        feats = np.hstack([feats, ce_scores.reshape(-1, 1).astype(np.float32)])
        print(f"  -> ce_score attached ({time.time() - t0:.1f}s)")
    else:
        print("[3/4] No ce_model/ directory — running with string features only.")

    print("[4/4] Predicting and saving...")
    clf = joblib.load(MODEL_PATH)
    if clf.n_features_in_ != feats.shape[1]:
        raise ValueError(
            f"model expects {clf.n_features_in_} features but got {feats.shape[1]}; "
            f"the GBM and the ce_model/ directory are out of sync"
        )
    predictions = clf.predict_proba(feats)[:, 1]

    results_df = pd.DataFrame({
        "id1": match_df["id1"].values,
        "id2": match_df["id2"].values,
        "predict": predictions,
    })
    results_df.to_csv(args.output_path, index=False)
    print(f"Done: {len(results_df)} rows in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
