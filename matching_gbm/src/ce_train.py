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
from src.pipeline import load_item_texts, load_items_by_id
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


def make_loaders(train_ds, val_ds, tokenizer, args, device):
    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader


def train_stage(model, tokenizer, train_loader, val_loader, val_labels,
                device, amp_dtype, epochs, lr, warmup_ratio, stage, save_dir=None):
    """Runs one training stage, always scoring on the same human validation set.

    Validation is human-labeled even during LLM pretraining: the point of that
    stage is what it does for human-label performance, and tracking agreement
    with the LLM labeler instead would measure the wrong thing entirely.
    """
    steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * warmup_ratio), steps
    )
    # Soft targets pass through unchanged: BCEWithLogitsLoss accepts any target
    # in [0,1], and for an LLM score of t it is minimized at sigmoid(z)=t. A
    # maximally unsure t=0.5 therefore contributes almost no gradient in either
    # direction on its own — the uncertainty is handled by the loss rather than
    # by an external weighting term.
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"[{stage}] {len(train_loader)} steps/epoch x {epochs} epochs, lr={lr:g}")
    best_auc = -1.0
    for epoch in range(epochs):
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
                print(f"  [{stage}] epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f} {rate:.0f} pairs/s")

        val_scores = predict_scores(model, val_loader, device, amp_dtype)
        auc = roc_auc_score(val_labels, val_scores)
        ap = average_precision_score(val_labels, val_scores)
        print(f"  [{stage}] epoch {epoch}: human-val ROC-AUC={auc:.4f} PR-AUC={ap:.4f}")

        if auc > best_auc:
            best_auc = auc
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"  [{stage}] saved to {save_dir} (best so far)")

    return best_auc


def load_pretrain_pairs(args, exclude_items):
    """LLM-labeled pairs for pretraining, optionally restricted to categories.

    Targeting categories is the point rather than an optimization: measured
    per-category ROC-AUC runs from 0.67 (Автотовары) to 0.95 (Аптека), and the
    weakest categories are exactly where the LLM file has the most coverage —
    Автотовары holds 986k items there against 38k in the human file. A random
    sample of 11M pairs would spend most of its budget on categories that are
    already strong.
    """
    match_df = pd.read_parquet(args.pretrain_matches_path)

    if args.pretrain_categories:
        print("  reading item categories...")
        cats_df = pd.read_parquet(args.pretrain_items_path, columns=["id", "category"])
        wanted = {c.strip() for c in args.pretrain_categories.split(",")}
        missing = wanted - set(cats_df["category"].unique())
        if missing:
            raise ValueError(f"unknown categories: {sorted(missing)}")
        keep_ids = set(cats_df[cats_df["category"].isin(wanted)]["id"].values)
        del cats_df
        print(f"  -> {len(keep_ids)} items in {sorted(wanted)}")

        in_scope = match_df["id1"].isin(keep_ids) & match_df["id2"].isin(keep_ids)
        match_df = match_df[in_scope]
        del keep_ids
        print(f"  -> {len(match_df)} pairs within scope")

    # An item held out for human validation must not appear in pretraining
    # either, or the fine-tuned model has effectively seen the val pairs.
    touches_val = match_df["id1"].isin(exclude_items) | match_df["id2"].isin(exclude_items)
    match_df = match_df[~touches_val]

    # t == 0.5 carries no directional signal at all; drop rather than train on it.
    match_df = match_df[match_df["target"] != 0.5]

    if args.max_pretrain_pairs and len(match_df) > args.max_pretrain_pairs:
        match_df = match_df.sample(
            n=args.max_pretrain_pairs, random_state=1234
        )
    match_df = match_df.reset_index(drop=True)
    print(f"  -> {len(match_df)} pretraining pairs after filtering")

    needed = set(match_df["id1"].values) | set(match_df["id2"].values)
    print(f"  loading texts for {len(needed)} items...")
    texts = load_item_texts(args.pretrain_items_path, needed, max_attrs=args.max_attrs)
    known = texts.__contains__
    usable = np.fromiter(
        (known(a) and known(b) for a, b in zip(match_df["id1"].values, match_df["id2"].values)),
        dtype=bool, count=len(match_df),
    )
    match_df = match_df[usable].reset_index(drop=True)
    print(f"  -> {len(match_df)} pairs with both items present")
    return match_df, texts


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

    parser.add_argument("--pretrain_matches_path", default=None,
                        help="LLM-labeled matches to pretrain on before fine-tuning")
    parser.add_argument("--pretrain_items_path", default="items.parquet")
    parser.add_argument("--pretrain_categories", default=None,
                        help="comma-separated categories to restrict pretraining to")
    parser.add_argument("--max_pretrain_pairs", type=int, default=1_000_000)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--pretrain_lr", type=float, default=3e-5)
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

    val_items = set(match_df.iloc[val_idx]["id1"]) | set(match_df.iloc[val_idx]["id2"])
    train_ds = PairDataset(
        [texts1[i] for i in train_idx], [texts2[i] for i in train_idx], labels[train_idx]
    )
    val_ds = PairDataset(
        [texts1[i] for i in val_idx], [texts2[i] for i in val_idx], labels[val_idx]
    )
    val_labels = labels[val_idx]

    print("[4/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1
    ).to(device)

    if args.pretrain_matches_path:
        print("[4.5/5] Building LLM pretraining set...")
        pre_df, pre_item_texts = load_pretrain_pairs(args, val_items)
        # Lists of references into the same text objects — two pointers per
        # pair, not two copies of the text.
        pre_texts1 = [pre_item_texts[i] for i in pre_df["id1"].values]
        pre_texts2 = [pre_item_texts[i] for i in pre_df["id2"].values]
        del pre_item_texts
        # Soft targets, not binarized: t is the labeler's confidence, and
        # rounding it to 0/1 throws that away right where it is most useful.
        pre_labels = pre_df["target"].values.astype(np.float32)
        pre_ds = PairDataset(pre_texts1, pre_texts2, pre_labels)

        pre_loader, val_loader = make_loaders(pre_ds, val_ds, tokenizer, args, device)
        train_stage(
            model, tokenizer, pre_loader, val_loader, val_labels, device, amp_dtype,
            epochs=args.pretrain_epochs, lr=args.pretrain_lr,
            warmup_ratio=args.warmup_ratio, stage="pretrain", save_dir=None,
        )
        del pre_ds, pre_loader, pre_texts1, pre_texts2, pre_df

    print("[5/5] Fine-tuning on human labels...")
    train_loader, val_loader = make_loaders(train_ds, val_ds, tokenizer, args, device)
    best_auc = train_stage(
        model, tokenizer, train_loader, val_loader, val_labels, device, amp_dtype,
        epochs=args.epochs, lr=args.lr, warmup_ratio=args.warmup_ratio,
        stage="finetune", save_dir=args.output_dir,
    )
    print(f"Best human-val ROC-AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
