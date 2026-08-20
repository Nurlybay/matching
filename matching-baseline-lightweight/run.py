import argparse
import os

import sys

import pandas as pd
import numpy as np

from src.utils import predict_pipeline

CLASSIFIER_PATH = "baseline_logreg_l12.joblib"
MODEL_CE_PATH = "models/cross-encoder-ms-marco-MiniLM-L12-v2"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    args = parser.parse_args()

    items_path = args.items_path
    matches_path = args.matches_path
    output_path = args.output_path

    # full pipeline in one go
    predictions_df = predict_pipeline(
        data_path=items_path,
        match_path=matches_path,
        model_path=MODEL_CE_PATH,
        logreg_path=CLASSIFIER_PATH,
        output_csv_path=output_path,
        batch_size=512
    )



if __name__ == "__main__":
    main()
