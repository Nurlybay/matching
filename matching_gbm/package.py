"""Assembles a submission archive.

Run from the matching_gbm directory once the model files are in place:

    python3 package.py --gbm gbm_model_v4.joblib --ce ce_model_v3 --out submission.zip

Checks the archive is self-contained and runnable before zipping, because the
evaluation environment has no network and reports nothing but "failed".
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Only what run.py actually imports. src/ce_train.py, src/train.py and
# src/analyze.py are training-time tools and are deliberately left out — they
# pull in sklearn.inspection and would bloat the archive for no runtime use.
RUNTIME_SOURCES = [
    "run.py",
    "metadata.json",
    os.path.join("src", "features.py"),
    os.path.join("src", "pipeline.py"),
    os.path.join("src", "model_io.py"),
    os.path.join("src", "ce_score.py"),
]

MAX_ARCHIVE_MB = 5000


def human_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def build_tree(staging, gbm_path, ce_dir):
    for rel in RUNTIME_SOURCES:
        src = os.path.join(HERE, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing required file: {rel}")
        dst = os.path.join(staging, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    shutil.copy2(gbm_path, os.path.join(staging, "gbm_model.joblib"))

    if ce_dir:
        dst_ce = os.path.join(staging, "ce_model")
        # The tokenizer and config are as essential as the weights; copying the
        # whole directory avoids reasoning about which files transformers needs.
        shutil.copytree(ce_dir, dst_ce)
        for name in os.listdir(dst_ce):
            print(f"    ce_model/{name}")


def smoke_test(staging, items_path, matches_path, n_pairs):
    """Runs the packaged tree against a small slice, the way the check stage
    will. Catching an import or path error here costs seconds; catching it
    after submission costs the whole attempt."""
    import pandas as pd

    tmp_matches = os.path.join(staging, "_smoke_matches.parquet")
    df = pd.read_parquet(matches_path).head(n_pairs)
    df[["id1", "id2"]].to_parquet(tmp_matches)

    out = os.path.join(staging, "_smoke_out.csv")
    cmd = [
        sys.executable, "run.py",
        "--items_path", os.path.abspath(items_path),
        "--matches_path", tmp_matches,
        "--output_path", out,
    ]
    print(f"  running: {' '.join(cmd[:3])} ... ({n_pairs} pairs)")
    result = subprocess.run(cmd, cwd=staging, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("smoke test failed — the archive would fail the docker stage")

    written = pd.read_csv(out)
    if list(written.columns) != ["id1", "id2", "predict"]:
        raise RuntimeError(f"wrong columns: {list(written.columns)}")
    if len(written) != len(df):
        raise RuntimeError(f"{len(written)} rows written for {len(df)} input pairs")
    if written["predict"].isna().any():
        raise RuntimeError("predict column contains NaN")

    print(f"  ok: {len(written)} rows, columns {list(written.columns)}")
    os.remove(tmp_matches)
    os.remove(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbm", default="gbm_model.joblib", help="trained GBM bundle")
    parser.add_argument("--ce", default="ce_model", help="cross-encoder dir ('' to omit)")
    parser.add_argument("--out", default="submission.zip")
    parser.add_argument("--items_path", default=None, help="items parquet for the smoke test")
    parser.add_argument("--matches_path", default=None, help="matches parquet for the smoke test")
    parser.add_argument("--smoke_pairs", type=int, default=1000,
                        help="matches the 1000-pair check stage")
    args = parser.parse_args()

    gbm_path = os.path.abspath(args.gbm)
    if not os.path.exists(gbm_path):
        raise FileNotFoundError(gbm_path)
    ce_dir = os.path.abspath(args.ce) if args.ce else None
    if ce_dir and not os.path.isdir(ce_dir):
        raise FileNotFoundError(ce_dir)

    with tempfile.TemporaryDirectory() as staging:
        print("[1/3] Staging files...")
        build_tree(staging, gbm_path, ce_dir)
        for rel in RUNTIME_SOURCES:
            print(f"    {rel}")
        print("    gbm_model.joblib")

        if args.items_path and args.matches_path:
            print("[2/3] Smoke test (as the check stage would run it)...")
            smoke_test(staging, args.items_path, args.matches_path, args.smoke_pairs)
        else:
            print("[2/3] Smoke test SKIPPED — pass --items_path and --matches_path "
                  "to actually verify the archive runs.")

        print("[3/3] Zipping...")
        out_path = os.path.abspath(args.out)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(staging):
                for name in files:
                    full = os.path.join(root, name)
                    zf.write(full, os.path.relpath(full, staging))

    size_mb = human_mb(os.path.getsize(out_path))
    print(f"\n{out_path}  ({size_mb:.1f} MB)")
    if size_mb > MAX_ARCHIVE_MB:
        print(f"WARNING: over the {MAX_ARCHIVE_MB} MB archive limit", file=sys.stderr)


if __name__ == "__main__":
    main()
