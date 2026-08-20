import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.features import item_text, prepare_item
from src.pipeline import load_items_by_id
from src.split import human_train_val_indices, valid_pairs_mask

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class PairDataset(Dataset):
    """Holds texts, not token ids — tokenization happens in the collate_fn so
    each batch pads to its own longest sequence instead of a global max."""

    def __init__(self, texts1, texts2, labels):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.texts1[i], self.texts2[i], self.labels[i]


def make_collate(tokenizer, max_length):
    def collate(batch):
        t1, t2, labels = zip(*batch)
        enc = tokenizer(
            list(t1), list(t2),
            padding=True, truncation="longest_first",
            max_length=max_length, return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        return enc

    return collate


@torch.no_grad()
def predict_scores(model, loader, device, amp_dtype):
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**batch).logits.squeeze(-1)
        out.append(logits.float().cpu())
    return torch.cat(out).numpy()


def build_pair_texts(match_df, items_by_id, max_attrs):
    """Materializes one text per item actually used, then indexes pairs into it —
    an item appearing in several pairs is only formatted once."""
    text_cache = {}

    def get_text(item_id):
        text = text_cache.get(item_id)
        if text is None:
            text = item_text(prepare_item(item_id, *items_by_id[item_id]), max_attrs=max_attrs)
            text_cache[item_id] = text
        return text

    texts1 = [get_text(i) for i in match_df["id1"].values]
    texts2 = [get_text(i) for i in match_df["id2"].values]
    return texts1, texts2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_human_path", default="items_human.parquet")
    parser.add_argument("--matches_path", default="matches.parquet")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default="ce_model")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--max_attrs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_pairs", type=int, default=0,
                        help="cap training pairs for a quick smoke run (0 = use all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 needs Ampere+; older cards fall back to fp16, CPU stays fp32.
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={device} amp_dtype={amp_dtype} model={args.model_name}")

    print("[1/5] Loading items and matches...")
    items_by_id = load_items_by_id(args.items_human_path)
    match_df = pd.read_parquet(args.matches_path)
    match_df = match_df[valid_pairs_mask(match_df, items_by_id)].reset_index(drop=True)
    print(f"  -> {len(match_df)} usable pairs")

    print("[2/5] Group-aware split (identical to the GBM's, see src/split.py)...")
    train_idx, val_idx, groups = human_train_val_indices(
        match_df["id1"].values, match_df["id2"].values
    )
    print(f"  -> {len(np.unique(groups))} components; train={len(train_idx)} val={len(val_idx)}")

    print("[3/5] Building texts...")
    texts1, texts2 = build_pair_texts(match_df, items_by_id, args.max_attrs)
    labels = (match_df["target"].values >= 0.5).astype(np.float32)
    del items_by_id

    if args.max_train_pairs:
        train_idx = train_idx[: args.max_train_pairs]
        print(f"  -> capped train to {len(train_idx)} pairs")

    train_ds = PairDataset(
        [texts1[i] for i in train_idx], [texts2[i] for i in train_idx], labels[train_idx]
    )
    val_ds = PairDataset(
        [texts1[i] for i in val_idx], [texts2[i] for i in val_idx], labels[val_idx]
    )

    print("[4/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1
    ).to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * args.warmup_ratio), steps
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"[5/5] Training: {len(train_loader)} steps/epoch x {args.epochs} epochs")
    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for step, batch in enumerate(train_loader, 1):
            target = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits.float(), target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item() * len(target)
            seen += len(target)
            if step % 200 == 0:
                rate = seen / (time.time() - t0)
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f} {rate:.0f} pairs/s")

        val_scores = predict_scores(model, val_loader, device, amp_dtype)
        auc = roc_auc_score(labels[val_idx], val_scores)
        ap = average_precision_score(labels[val_idx], val_scores)
        print(f"  epoch {epoch}: human-val ROC-AUC={auc:.4f} PR-AUC={ap:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  saved to {args.output_dir} (best so far)")

    print(f"Best human-val ROC-AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
