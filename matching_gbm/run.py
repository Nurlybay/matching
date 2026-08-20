import argparse

import joblib
import pandas as pd

from src.pipeline import build_feature_matrix, load_items_by_id

MODEL_PATH = "gbm_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    args = parser.parse_args()

    print("[1/4] Loading items...")
    items_by_id = load_items_by_id(args.items_path)

    print("[2/4] Building features...")
    match_df = pd.read_parquet(args.matches_path)
    feats = build_feature_matrix(match_df, items_by_id, n_jobs=1)

    print("[3/4] Loading model and predicting...")
    clf = joblib.load(MODEL_PATH)
    predictions = clf.predict_proba(feats)[:, 1]

    print(f"[4/4] Saving predictions to {args.output_path}...")
    results_df = pd.DataFrame({
        "id1": match_df["id1"].values,
        "id2": match_df["id2"].values,
        "predict": predictions,
    })
    results_df.to_csv(args.output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
