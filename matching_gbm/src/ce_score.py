import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features import item_text, prepare_item
from src.pipeline import load_items_by_id


def load_ce(model_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    return model, tokenizer, device, amp_dtype


@torch.inference_mode()
def score_pairs(model, tokenizer, device, amp_dtype, texts1, texts2,
                batch_size=256, max_length=192, show_every=0):
    """Logit per pair, in input order.

    Pairs are processed shortest-first (bucketed by character length as a cheap
    proxy for token count) so each batch pads to a similar length instead of
    dragging short pairs up to the longest in a random batch — then written
    back through the sort permutation so the output stays aligned with input.
    """
    n = len(texts1)
    scores = np.empty(n, dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(texts1, texts2)])

    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [texts1[i] for i in idx], [texts2[i] for i in idx],
            padding=True, truncation="longest_first",
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**enc).logits.squeeze(-1)
        scores[idx] = logits.float().cpu().numpy()

        if show_every and (start // batch_size) % show_every == 0:
            print(f"  scored {min(start + batch_size, n)}/{n}")

    return scores


def build_pair_texts_aligned(match_df, items_by_id, max_attrs=12):
    """Texts for every row of match_df, plus a mask of rows whose items were
    both found. Rows with a missing item still get an entry (empty text) so
    the caller can keep a 1:1 alignment with match_df and fill NaN later."""
    text_cache = {}

    def get_text(item_id):
        if item_id in text_cache:
            return text_cache[item_id]
        raw = items_by_id.get(item_id)
        text = "" if raw is None else item_text(prepare_item(item_id, *raw), max_attrs=max_attrs)
        text_cache[item_id] = text
        return text

    texts1 = [get_text(i) for i in match_df["id1"].values]
    texts2 = [get_text(i) for i in match_df["id2"].values]
    known = np.fromiter(
        ((a != "") and (b != "") for a, b in zip(texts1, texts2)),
        dtype=bool, count=len(match_df),
    )
    return texts1, texts2, known


def score_match_df(match_df, items_by_id, model_dir, batch_size=256,
                   max_length=192, max_attrs=12, show_every=0):
    """CE logit per row of match_df, NaN where an item is missing."""
    model, tokenizer, device, amp_dtype = load_ce(model_dir)
    texts1, texts2, known = build_pair_texts_aligned(match_df, items_by_id, max_attrs)

    scores = np.full(len(match_df), np.nan, dtype=np.float32)
    if known.any():
        idx = np.flatnonzero(known)
        scores[idx] = score_pairs(
            model, tokenizer, device, amp_dtype,
            [texts1[i] for i in idx], [texts2[i] for i in idx],
            batch_size=batch_size, max_length=max_length, show_every=show_every,
        )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", required=True)
    parser.add_argument("--matches_path", required=True)
    parser.add_argument("--ce_model_dir", default="ce_model")
    parser.add_argument("--output_path", required=True, help="where to write the .npy scores")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--max_attrs", type=int, default=12)
    args = parser.parse_args()

    print("[1/3] Loading items...")
    items_by_id = load_items_by_id(args.items_path)

    print("[2/3] Scoring pairs...")
    match_df = pd.read_parquet(args.matches_path)
    scores = score_match_df(
        match_df, items_by_id, args.ce_model_dir,
        batch_size=args.batch_size, max_length=args.max_length,
        max_attrs=args.max_attrs, show_every=50,
    )

    print(f"[3/3] Saving {len(scores)} scores to {args.output_path}...")
    np.save(args.output_path, scores)
    print(f"  NaN (missing item): {int(np.isnan(scores).sum())}")


if __name__ == "__main__":
    main()
