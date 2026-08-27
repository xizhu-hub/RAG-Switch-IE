# -*- coding: utf-8 -*-
import os
import re
import json
import math
import argparse
from typing import List, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# DS model
# =========================

class DsRegressor(nn.Module):
    def __init__(self, in_dim=3, width=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width, width), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class El2nBackbone(nn.Module):
    def __init__(self, ds_width, dropout, stat_dim=3):
        super().__init__()
        self.ds_head = DsRegressor(in_dim=stat_dim, width=ds_width, drop=dropout)

    def forward(self, input_ids, mask, stat3):
        return self.ds_head(stat3)


# =========================
# Tools
# =========================

def replace_el2n_in_text(text: str, new_val: float) -> str:
    pattern = r"(EL2N score:\s*)(-?\d+(?:\.\d+)?)"
    return re.sub(pattern, lambda m: f"{m.group(1)}{new_val:.3f}", text, count=1)


def normalize_ground_truth(s: str) -> int:
    s = str(s).strip().lower()
    return 1 if "yes" in s else 0


def get_first_token_ids(tokenizer, variants: List[str]) -> List[int]:
    ids = set()
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) > 0:
            ids.add(enc[0])
    return sorted(list(ids))


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_metrics(labels: List[int], probs: List[float], threshold: float):
    preds = [1 if p > threshold else 0 for p in probs]

    tp = sum((t == 1 and p == 1) for t, p in zip(labels, preds))
    fp = sum((t == 0 and p == 1) for t, p in zip(labels, preds))
    tn = sum((t == 0 and p == 0) for t, p in zip(labels, preds))
    fn = sum((t == 1 and p == 0) for t, p in zip(labels, preds))

    acc = safe_div(tp + tn, len(labels))

    p_y = safe_div(tp, tp + fp)
    r_y = safe_div(tp, tp + fn)
    f1_y = safe_div(2 * p_y * r_y, p_y + r_y)

    p_n = safe_div(tn, tn + fn)
    r_n = safe_div(tn, tn + fp)
    f1_n = safe_div(2 * p_n * r_n, p_n + r_n)

    macro_f1 = (f1_y + f1_n) / 2
    bal_acc = (r_y + r_n) / 2

    return {
        "threshold": threshold,
        "acc": acc,
        "micro_f1_yes": f1_y,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "yes_precision": p_y,
        "yes_recall": r_y,
        "yes_f1": f1_y,
        "no_precision": p_n,
        "no_recall": r_n,
        "no_f1": f1_n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def search_best_threshold(labels: List[int], probs: List[float]):
    best = None
    for thr in np.arange(0.1, 0.9, 0.01):
        m = compute_metrics(labels, probs, float(thr))
        if best is None or m["micro_f1_yes"] > best["micro_f1_yes"]:
            best = m
    return best


# =========================
# Dataset
# =========================

class TestQADataset(TorchDataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_src: int,
        ds_model=None,
        device=None,
        stat_key: str = "stat_vector",
    ):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.ds_model = ds_model
        self.device = device
        self.stat_key = stat_key

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                uid = obj.get("global_uid", "")
                text = obj.get("input_text", "")

                if self.ds_model is not None and self.stat_key in obj:
                    stat = obj.get(self.stat_key, [0.0, 0.0, 0.0])[:3]
                    stat = torch.tensor(stat, dtype=torch.float32).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        pred_el2n = self.ds_model(None, None, stat).item()

                    text = replace_el2n_in_text(text, pred_el2n)

                if "label" in obj:
                    label = int(obj["label"])
                else:
                    label = normalize_ground_truth(obj.get("ground_truth", "no"))

                self.rows.append({
                    "uid": uid,
                    "input_text": text,
                    "label": label,
                    "ground_truth": obj.get("ground_truth", ""),
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tokenizer(
            r["input_text"],
            max_length=self.max_src,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        return {
            "uid": r["uid"],
            "input_text": r["input_text"],
            "label": torch.tensor(r["label"], dtype=torch.long),
            "ground_truth": r["ground_truth"],
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


# =========================
Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage2_model_dir", type=str, required=True,
                        help="e.g. out_dir/stage2_best")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="test sets path")
    parser.add_argument("--out_pred_jsonl", type=str, default="",
                        help="optional")

    parser.add_argument("--id_max_source_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)

    parser.add_argument("--threshold", type=float, default=None,
                        help="use threshold or 0.5")
    parser.add_argument("--search_threshold", action="store_true",
                        help="for test")

    parser.add_argument("--use_ds_el2n", action="store_true",
                        help="reload phase1 DS model")
    parser.add_argument("--ds_ckpt_dir", type=str, default="",
                        help="path ofr phase1 DS model")
    parser.add_argument("--ds_width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stat_key", type=str, default="stat_vector")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.stage2_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.stage2_model_dir).to(device)
    model.eval()

    threshold = args.threshold
    eval_info_path = os.path.join(args.stage2_model_dir, "eval_info.json")

    if threshold is None:
        if os.path.exists(eval_info_path):
            with open(eval_info_path, "r", encoding="utf-8") as f:
                eval_info = json.load(f)
            threshold = float(eval_info.get("threshold", 0.5))
            print(f"[INFO] Loaded threshold from eval_info.json: {threshold:.4f}")
        else:
            threshold = 0.5
            print("[INFO] No threshold provided. Use default threshold = 0.5")

    ds_model = None
    if args.use_ds_el2n:
        if not args.ds_ckpt_dir:
            raise ValueError("--use_ds_el2n requires --ds_ckpt_dir")

        ckpt_path = os.path.join(args.ds_ckpt_dir, "model_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Cannot find DS checkpoint: {ckpt_path}")

        ds_model = El2nBackbone(args.ds_width, args.dropout).to(device)
        ds_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        ds_model.eval()

        print(f"[INFO] Loaded DS model from {ckpt_path}")
        print("[INFO] Test input_text EL2N score will be replaced by predicted DS EL2N.")

    test_ds = TestQADataset(
        path=args.test_jsonl,
        tokenizer=tokenizer,
        max_src=args.id_max_source_len,
        ds_model=ds_model,
        device=device,
        stat_key=args.stat_key,
    )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    yes_token_ids = get_first_token_ids(tokenizer, ["yes", "Yes"])
    no_token_ids = get_first_token_ids(tokenizer, ["no", "No"])

    print(f"[INFO] yes_token_ids = {yes_token_ids}")
    print(f"[INFO] no_token_ids  = {no_token_ids}")

    all_probs = []
    all_labels = []
    all_uids = []
    all_texts = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                model.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            logits = outputs.logits[:, 0, :]
            probs = F.softmax(logits, dim=-1)

            p_yes = probs[:, yes_token_ids].sum(dim=-1)
            p_no = probs[:, no_token_ids].sum(dim=-1)

            norm_p_yes = p_yes / (p_yes + p_no + 1e-8)

            all_probs.extend(norm_p_yes.detach().cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_uids.extend(batch["uid"])
            all_texts.extend(batch["input_text"])
            all_gts.extend(batch["ground_truth"])

    metrics = compute_metrics(all_labels, all_probs, threshold)

    print("\n========== Test Result ==========")
    print(f"Samples: {len(all_labels)}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Micro-F1 / YES-F1: {metrics['micro_f1_yes']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_acc']:.4f}")

    print("\n--- YES class ---")
    print(f"P={metrics['yes_precision']:.4f} R={metrics['yes_recall']:.4f} F1={metrics['yes_f1']:.4f}")

    print("\n--- NO class ---")
    print(f"P={metrics['no_precision']:.4f} R={metrics['no_recall']:.4f} F1={metrics['no_f1']:.4f}")

    print("\n--- Confusion Matrix ---")
    print(f"tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']}")

    if args.search_threshold:
        best = search_best_threshold(all_labels, all_probs)

        print("\n========== Best Threshold on Test Set ==========")
        print(f"Best threshold: {best['threshold']:.4f}")
        print(f"Accuracy: {best['acc']:.4f}")
        print(f"Micro-F1 / YES-F1: {best['micro_f1_yes']:.4f}")
        print(f"Macro-F1: {best['macro_f1']:.4f}")
        print(f"Balanced Accuracy: {best['balanced_acc']:.4f}")
        print(f"tp={best['tp']} fp={best['fp']} tn={best['tn']} fn={best['fn']}")

    if args.out_pred_jsonl:
        with open(args.out_pred_jsonl, "w", encoding="utf-8") as f:
            for uid, text, gt, label, prob in zip(all_uids, all_texts, all_gts, all_labels, all_probs):
                pred = 1 if prob > threshold else 0
                obj = {
                    "global_uid": uid,
                    "label": int(label),
                    "pred": int(pred),
                    "prob_yes": float(prob),
                    "threshold": float(threshold),
                    "ground_truth": gt,
                    "correct": bool(pred == label),
                    "input_text": text,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"\n[INFO] Prediction file saved to: {args.out_pred_jsonl}")


if __name__ == "__main__":
    main()
