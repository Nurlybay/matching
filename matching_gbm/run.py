import argparse
import os
import sys
import time

import pandas as pd

from src.model_io import load_bundle
from src.pipeline import build_full_features, load_items_by_id

# Paths are relative to this file, not to the working directory the harness
# happens to launch us from.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "gbm_model.joblib")
CE_MODEL_DIR = os.path.join(HERE, "ce_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # The task description writes --output-path with a hyphen while the
    # reference baseline uses --output_path with an underscore. argparse does
    # not treat those as equivalent — it only normalizes the *dest* — so a
    # solution accepting one spelling dies instantly on a harness passing the
    # other. Accept every documented form.
    parser.add_argument("--items_path", "--items-path", "-i",
                        dest="items_path", required=True,
                        help="items parquet")
    parser.add_argument("--matches_path", "--matches-path", "-m",
                        dest="matches_path", required=True,
                        help="candidate pairs parquet")
    parser.add_argument("--output_path", "--output-path", "-o",
                        dest="output_path", required=True,
                        help="destination csv")
    parser.add_argument("--ce_batch_size", type=int, default=256)
    parser.add_argument("--ce_max_length", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t0 = time.time()
    use_ce = os.path.isdir(CE_MODEL_DIR)

    print(f"[1/4] Loading items and pairs...", flush=True)
    items_by_id = load_items_by_id(args.items_path)
    match_df = pd.read_parquet(args.matches_path)
    print(f"  -> {len(items_by_id)} items, {len(match_df)} pairs "
          f"({time.time() - t0:.1f}s)", flush=True)

    # The category map travels with the model rather than being rebuilt from
    # the test items: the test set holds different products, and renumbering
    # the categories would feed the model codes meaning something other than
    # they did at fit time.
    clf, category_map, _ = load_bundle(MODEL_PATH)

    ce_scores = None
    if use_ce:
        # Imported lazily so a GBM-only submission never pays the torch import
        # or CUDA init, which matter against the 1-minute check-stage limit.
        from src.ce_score import score_match_df

        print("[2/4] Scoring pairs with the cross-encoder...", flush=True)
        ce_scores = score_match_df(
            match_df, items_by_id, CE_MODEL_DIR,
            batch_size=args.ce_batch_size, max_length=args.ce_max_length,
        )
        print(f"  -> scored ({time.time() - t0:.1f}s)", flush=True)
    else:
        print("[2/4] No ce_model/ directory — string features only.", flush=True)

    print("[3/4] Building features...", flush=True)
    feats = build_full_features(
        match_df, items_by_id, category_map=category_map, ce_scores=ce_scores, n_jobs=1
    )
    print(f"  -> {feats.shape} ({time.time() - t0:.1f}s)", flush=True)

    print("[4/4] Predicting and saving...", flush=True)
    if clf.n_features_in_ != feats.shape[1]:
        raise ValueError(
            f"model expects {clf.n_features_in_} features but got {feats.shape[1]}; "
            f"gbm_model.joblib and ce_model/ are out of sync"
        )
    predictions = clf.predict_proba(feats)[:, 1]

    # Every input pair must appear in the output, in the same order and with
    # the ids exactly as given — the result stage rejects a file that misses
    # even one pair.
    results_df = pd.DataFrame({
        "id1": match_df["id1"].values,
        "id2": match_df["id2"].values,
        "predict": predictions,
    })
    if len(results_df) != len(match_df):
        raise AssertionError(
            f"produced {len(results_df)} rows for {len(match_df)} input pairs"
        )

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.output_path, index=False)
    print(f"Done: {len(results_df)} rows in {time.time() - t0:.1f}s total.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # The harness only reports that the container failed, so make sure the
        # traceback reaches stdout/stderr before the non-zero exit.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
