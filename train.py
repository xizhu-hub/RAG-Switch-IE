# -*- coding: utf-8 -*-
import os
import re
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============ Config ============

@dataclass
class Args:
    mode: str
    t5_path: str
    out_dir: str
    data_path: str = ""
    phase1_data_path: str = ""
    val_jsonl: str = ""
    init_from: str = ""
    el2n_key: str = "el2n_z"
    stat_key: str = "stat_vector"
    val_ratio: float = 0.1
    pool: str = "mean"
    max_length: int = 512
    hidden_dim: int = 256
    ds_width: int = 256
    dropout: float = 0.2
    learning_rate_ds: float = 5e-4
    weight_decay_ds: float = 0.05
    num_train_epochs: int = 50
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 12
    patience: int = 15
    id_model_name_or_path: str = "t5-large"
    id_max_source_len: int = 512
    id_max_target_len: int = 16
    label_smoothing: float = 0.0
    learning_rate_id: float = 3e-5
    weight_decay_id: float = 0.0
    select_metric: str = "micro_f1"
    no_f1_drop_tol: float = 0.0
    min_precision_no: float = -1.0
    min_recall_no: float = -1.0
    seed: int = 42
    val_jsonl: str = ""

# ============ Tools ============

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    if s.endswith("."): s = s[:-1]
    return "yes" if "yes" in s else "no"

def get_label_dist(rows: List[Dict]):
    labels = [int(r.get("label", 0)) for r in rows]
    counts = Counter(labels)
    total = len(labels) if len(labels) > 0 else 1
    dist_str = " | ".join([f"Label {k}: {v} ({v/total:.1%})" for k, v in sorted(counts.items())])
    return dist_str

def split_train_val(rows: List[Dict], test_size=0.1, seed=42):
    labels = [int(r.get("label", 0)) for r in rows]
    idx0 = [i for i, l in enumerate(labels) if l == 0]
    idx1 = [i for i, l in enumerate(labels) if l == 1]
    r = random.Random(seed)
    r.shuffle(idx0); r.shuffle(idx1)
    n0, n1 = int(len(idx0)*test_size), int(len(idx1)*test_size)
    val_idx = set(idx0[:n0] + idx1[:n1])
    return [i for i in range(len(rows)) if i not in val_idx], [i for i in range(len(rows)) if i in val_idx]

def oversample_to_majority(rows: List[Dict]) -> List[Dict]:
    by = {}
    for r in rows:
        y = int(r.get("label", -1))
        by.setdefault(y, []).append(r)
    if len(by) <= 1: return rows
    maxn = max(len(v) for v in by.values())
    out = []
    for y, lst in by.items():
        reps = math.ceil(maxn / len(lst))
        out.extend((lst * reps)[:maxn])
    random.shuffle(out)
    return out

# ============ 3. Model ============

class DsRegressor(nn.Module):
    def __init__(self, in_dim=3, width=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width, width), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class El2nBackbone(nn.Module):
    def __init__(self, ds_width, dropout, stat_dim=3):
        super().__init__()
        self.ds_head = DsRegressor(in_dim=stat_dim, width=ds_width, drop=dropout)
    def forward(self, input_ids, mask, stat3): return self.ds_head(stat3)

# ============ 4. Phase 1: DS Only ============

class JointDataset(TorchDataset):
    def __init__(self, path, el2n_key, stat_key):
        self.rows = []
        if not os.path.exists(path): return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                s = o.get(stat_key, [0.0, 0.0, 0.0])
                e = o.get(el2n_key, 0.0)
                self.rows.append({
                    "uid": o.get("global_uid",""), 
                    "stat3": torch.tensor(s[:3], dtype=torch.float32), 
                    "el2n": torch.tensor(float(e), dtype=torch.float32),
                    "label": o.get("label", 0)
                })
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def train_phase1_ds_only(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    full_ds = JointDataset(args.phase1_data_path, args.el2n_key, args.stat_key)
    tr_idx, va_idx = split_train_val(full_ds.rows, args.val_ratio, args.seed)
    
    print(f">>> Val Dist: {get_label_dist([full_ds.rows[i] for i in va_idx])}")

    train_loader = DataLoader([full_ds.rows[i] for i in tr_idx], batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader([full_ds.rows[i] for i in va_idx], batch_size=args.per_device_eval_batch_size)
    model = El2nBackbone(args.ds_width, args.dropout).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_ds, weight_decay=args.weight_decay_ds)
    best_mse = float('inf')
    patience_counter = 0
    for ep in range(1, args.num_train_epochs + 1):
        model.train(); train_loss_total = 0
        pbar = tqdm(train_loader, desc=f"DS Epoch {ep}")
        for batch in pbar:
            stat = batch["stat3"].to(device); el2n = batch["el2n"].to(device)
            pred = model(None, None, stat)
            loss = F.mse_loss(pred, el2n)
            if loss.dim() > 0: loss = loss.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_total += loss.item()
            pbar.set_postfix(loss=f"{train_loss_total/(pbar.n+1):.4f}")
        model.eval(); v_mse_sum, v_mae_sum, v_n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                stat = batch["stat3"].to(device); el2n = batch["el2n"].to(device)
                pred = model(None, None, stat)
                v_mse_sum += F.mse_loss(pred, el2n, reduction='sum').item()
                v_mae_sum += F.l1_loss(pred, el2n, reduction='sum').item()
                v_n += el2n.numel()
        avg_mse = v_mse_sum / max(1, v_n); avg_mae = v_mae_sum / max(1, v_n)
        print(f"[*] Epoch {ep} | Val MSE: {avg_mse:.6f} | Val MAE: {avg_mae:.6f}")
        if avg_mse < best_mse:
            best_mse = avg_mse; patience_counter = 0
            raw_model = model.module if hasattr(model, 'module') else model
            torch.save(raw_model.state_dict(), os.path.join(args.out_dir, "model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience: break

# ============ 5. Phase 2/3: ID training ============

class QADataset(TorchDataset):
    def __init__(self, rows, tokenizer, max_src, max_tgt):
        self.rows = rows; self.tok = tokenizer; self.max_src = max_src; self.max_tgt = max_tgt
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["input_text"], max_length=self.max_src, truncation=True, padding="max_length")
        lab = self.tok(text_target=r["ground_truth"], max_length=self.max_tgt, truncation=True, padding="max_length")
        target_ids = torch.tensor(lab["input_ids"])
        target_ids[target_ids == self.tok.pad_token_id] = -100
        return {"input_ids": torch.tensor(enc["input_ids"]), "attention_mask": torch.tensor(enc["attention_mask"]), "labels": target_ids}

def replace_el2n_in_text(text: str, new_val: float) -> str:
    pattern = r"(EL2N score:\s*)(-?\d+(?:\.\d+)?)"
    return re.sub(pattern, lambda m: f"{m.group(1)}{new_val:.3f}", text, count=1)

def train_phase2_3_id(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ds_model = El2nBackbone(args.ds_width, args.dropout).to(device)
    ckpt_path = os.path.join(args.init_from, "model_best.pt")
    ds_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    ds_model.eval()
    
    p1_dataset = JointDataset(args.phase1_data_path, args.el2n_key, args.stat_key)
    uid2stat = {r["uid"]: r["stat3"] for r in p1_dataset.rows if r["uid"]}
    
    def prepare_id_rows(path):
        if not path or not os.path.exists(path): return []
        processed = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                uid, txt = obj.get("global_uid", ""), obj.get("input_text", "")
                if uid in uid2stat:
                    with torch.no_grad():
                        pred_el2n = ds_model(None, None, uid2stat[uid].unsqueeze(0).to(device)).item()
                    txt = replace_el2n_in_text(txt, pred_el2n)
                processed.append({"input_text": txt, "ground_truth": obj.get("ground_truth", "no"), "label": int(obj.get("label", 0))})
        return processed

    print(">>> Preparing Data and Predicting EL2N...")
    all_rows = prepare_id_rows(args.data_path)
    val_rows = prepare_id_rows(args.val_jsonl) if args.val_jsonl else []
    if not val_rows:
        tr_idx, va_idx = split_train_val(all_rows, args.val_ratio, args.seed)
        train_rows, val_rows = [all_rows[i] for i in tr_idx], [all_rows[i] for i in va_idx]
    else: train_rows = all_rows

    print(f"--- Data Statistics ---")
    print(f"Original Train Set Dist: {get_label_dist(train_rows)}")
    print(f"Validation Set Dist: {get_label_dist(val_rows)}")
    
    train_rows = oversample_to_majority(train_rows)
    print(f"Oversampled Train Set Dist: {get_label_dist(train_rows)}")
    print(f"------------------------")

    tokenizer = AutoTokenizer.from_pretrained(args.id_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.id_model_name_or_path).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    train_loader = DataLoader(QADataset(train_rows, tokenizer, args.id_max_source_len, args.id_max_target_len), batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(QADataset(val_rows, tokenizer, args.id_max_source_len, args.id_max_target_len), batch_size=args.per_device_eval_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_id)
    
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]

    best_select_score = -1.0
    best_info = {}

    for ep in range(1, args.num_train_epochs + 1):
        model.train(); train_loss_total = 0
        pbar = tqdm(train_loader, desc=f"ID Epoch {ep}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if loss.dim() > 0: loss = loss.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_total += loss.item()
            pbar.set_postfix(loss=f"{train_loss_total/(pbar.n+1):.4f}")
        
        model.eval(); val_probs, val_labels = [], []
        with torch.no_grad():
            m = model.module if hasattr(model, 'module') else model
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                decoder_input_ids = torch.full((input_ids.size(0), 1), m.config.decoder_start_token_id, device=device)
                outputs = m(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                probs = F.softmax(logits, dim=-1)
                p_yes = probs[:, yes_token_id]
                p_no = probs[:, no_token_id]
                norm_p_yes = p_yes / (p_yes + p_no + 1e-8)
                val_probs.extend(norm_p_yes.cpu().tolist())
                target_ids = batch["labels"].clone(); target_ids[target_ids == -100] = tokenizer.pad_token_id
                val_labels.extend([1 if "yes" in tokenizer.decode(g, skip_special_tokens=True).lower() else 0 for g in target_ids])

        best_f1, best_thr = -1.0, 0.5
        for thr in np.arange(0.1, 0.9, 0.01):
            preds = [1 if p > thr else 0 for p in val_probs]
            tp = sum((t==1 and p==1) for t,p in zip(val_labels, preds))
            fp = sum((t==0 and p==1) for t,p in zip(val_labels, preds))
            fn = sum((t==1 and p==0) for t,p in zip(val_labels, preds))
            p_val = tp/(tp+fp) if (tp+fp)>0 else 0
            r_val = tp/(tp+fn) if (tp+fn)>0 else 0
            f1_val = 2*p_val*r_val/(p_val+r_val) if (p_val+r_val)>0 else 0
            if f1_val > best_f1: best_f1, best_thr = f1_val, thr

        final_preds = [1 if p > best_thr else 0 for p in val_probs]
        tp = sum((t==1 and p==1) for t,p in zip(val_labels, final_preds))
        fp = sum((t==0 and p==1) for t,p in zip(val_labels, final_preds))
        tn = sum((t==0 and p==0) for t,p in zip(val_labels, final_preds))
        fn = sum((t==1 and p==0) for t,p in zip(val_labels, final_preds))
        
        acc = (tp+tn)/len(val_labels); p_y = tp/(tp+fp) if (tp+fp)>0 else 0; r_y = tp/(tp+fn) if (tp+fn)>0 else 0; f1_y = 2*p_y*r_y/(p_y+r_y) if (p_y+r_y)>0 else 0
        p_n = tn/(tn+fn) if (tn+fn)>0 else 0; r_n = tn/(tn+fp) if (tn+fp)>0 else 0; f1_n = 2*p_n*r_n/(p_n+r_n) if (p_n+r_n)>0 else 0
        macro_f1 = (f1_y + f1_n)/2; bal_acc = (r_y + r_n)/2

        select_score = {
            "micro_f1": f1_y,
            "macro_f1": macro_f1,
            "bal_acc": bal_acc,
            "acc": acc,
        }.get(args.select_metric, f1_y)

        if select_score > best_select_score:
            best_select_score = select_score
            raw_model = model.module if hasattr(model, "module") else model

            save_dir = os.path.join(args.out_dir, "stage2_best")
            os.makedirs(save_dir, exist_ok=True)

            raw_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            best_info = {
                "epoch": ep,
                "threshold": float(best_thr),
                "select_metric": args.select_metric,
                "select_score": float(select_score),
                "acc": float(acc),
                "micro_f1_yes": float(f1_y),
                "macro_f1": float(macro_f1),
                "balanced_acc": float(bal_acc),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }

            with open(os.path.join(save_dir, "eval_info.json"), "w", encoding="utf-8") as f:
                json.dump(best_info, f, indent=2, ensure_ascii=False)

            print(f"[SAVE] Best stage2 model saved to {save_dir}")


        print(f"\n[*] ID Epoch {ep} | loss={train_loss_total/len(train_loader):.4f} | acc={acc:.4f} microF1={f1_y:.4f} macroF1={macro_f1:.4f} balAcc={bal_acc:.4f}")
        print(f"YES: P={p_y:.4f} R={r_y:.4f} F1={f1_y:.4f} | NO: P={p_n:.4f} R={r_n:.4f} F1={f1_n:.4f}")
        print(f"cm=[tp:{tp} fp:{fp} tn:{tn} fn:{fn}] | thr*={best_thr:.2f}")

# ============ 6. Args ============

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True)
    p.add_argument("--t5_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--phase1_data_path", type=str, default="")
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--init_from", type=str, default="")
    p.add_argument("--val_jsonl", type=str, default="")
    p.add_argument("--ds_width", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning_rate_ds", type=float, default=5e-4)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=12)
    p.add_argument("--id_model_name_or_path", type=str, default="t5-large")
    p.add_argument("--id_max_source_len", type=int, default=512)
    p.add_argument("--id_max_target_len", type=int, default=16)
    p.add_argument("--learning_rate_id", type=float, default=3e-5)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--el2n_key", type=str, default="el2n_z")
    p.add_argument("--stat_key", type=str, default="stat_vector")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--pool", type=str, default="mean")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--weight_decay_ds", type=float, default=0.05)
    p.add_argument("--weight_decay_id", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--select_metric", type=str, default="micro_f1")
    p.add_argument("--no_f1_drop_tol", type=float, default=0.0)
    p.add_argument("--min_precision_no", type=float, default=-1.0)
    p.add_argument("--min_recall_no", type=float, default=-1.0)
    return Args(**vars(p.parse_args()))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "ds_only": train_phase1_ds_only(args)
    elif args.mode in ["id_on_ds", "joint_lite"]: train_phase2_3_id(args)

if __name__ == "__main__": main()
