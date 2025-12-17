
# train_transformer_multilabel.py
import os
import json
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# -----------------------------
# Labels (keep fixed order)
# -----------------------------
LABELS = [
    "CONSULT","ASSAULT","DISAPPROVE","SUPPORT","COERCE","AGREE","AID","REJECT",
    "CONCEDE","COOPERATE","RETREAT","THREATHEN","DEMAND","PROTEST","SANCTION","MOBILIZE"
]
label2id = {l:i for i,l in enumerate(LABELS)}
N_CLASSES = len(LABELS)

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    csv_path: str = "../data/train_sample.csv"
    text_cols: Tuple[str, str] = ("title", "first_para")  # (title, first para). Set second to "" for title-only
    key_preference: Tuple[str, ...] = ("SOURCEURL", "GlobalEventID")  # article grouping key
    pretrained_model: str = "roberta-base"  # or "allenai/longformer-base-4096"
    max_length: int = 512                   # set to 1024/2048/4096 with Longformer
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    use_amp: bool = True
    use_focal: bool = False
    focal_gamma: float = 2.0
    dropout: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "outputs_transformer_multilabel"
    # early stopping
    es_patience: int = 2

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# -----------------------------
# Data aggregation
# -----------------------------
def aggregate_articles(df: pd.DataFrame,
                       text_cols: Tuple[str,str],
                       label_col: str,
                       key_preference: Tuple[str,...]) -> Tuple[pd.DataFrame, np.ndarray]:
    """One row per article; text = title + first_para; Y = multi-hot labels."""
    df[label_col] = df[label_col].astype(str).str.upper()
    df = df[df[label_col].isin(LABELS)].copy()

    # Article key
    key = None
    for k in key_preference:
        if k in df.columns and not df[k].isna().all():
            key = k; break
    if key is None:
        key = "__article_key__"
        df[key] = (df.get("title","").fillna("").astype(str) + "||" +
                   df.get("first_para","").fillna("").astype(str))

    # Build text feature
    col_a, col_b = text_cols
    text_a = df.get(col_a, "").fillna("").astype(str)
    text_b = df.get(col_b, "").fillna("").astype(str) if col_b else ""
    df["text_feat"] = (text_a + (" " + text_b if col_b else "")).str.strip()

    # Aggregate: unique labels per article + first text_feat (or longest)
    agg = (
        df.groupby(key)
          .agg({"text_feat": "first", label_col: lambda s: list(set(s))})
          .reset_index(drop=True)
    )

    # Multi-hot targets
    Y = np.zeros((len(agg), N_CLASSES), dtype=np.float32)
    for i, labs in enumerate(agg[label_col]):
        for lbl in labs:
            Y[i, label2id[lbl]] = 1.0

    return agg[["text_feat"]], Y

# -----------------------------
# Tokenized Dataset
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], targets: np.ndarray,
                 tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.targets = targets.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return item

# -----------------------------
# Model: Encoder + Classification head
# -----------------------------
class MultiLabelTransformer(nn.Module):
    def __init__(self, pretrained_model: str, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token or pooled output if available; fallback to mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            h = outputs.pooler_output                     # [B, H]
        else:
            # mean-pool last hidden state (mask-aware)
            last_hidden = outputs.last_hidden_state       # [B, T, H]
            mask = attention_mask.unsqueeze(-1)           # [B, T, 1]
            h = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(self.dropout(h))         # [B, C]

        return logits

# -----------------------------
# Losses
# -----------------------------
def effective_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Inverse-sqrt pos_weight from training positives per class."""
    counts = y_train.sum(axis=0)
    mx = float(np.max(counts))
    w = mx / np.sqrt(np.clip(counts, 1.0, None))
    w = np.clip(w, 1.0, None).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32)

def focal_bce_with_logits(logits, targets, alpha=None, gamma: float = 2.0, reduction="mean"):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=alpha, reduction="none"
    )
    pt = torch.exp(-bce)
    loss = ((1 - pt) ** gamma) * bce
    return loss.mean() if reduction == "mean" else loss.sum()

# -----------------------------
# Metrics & thresholds
# -----------------------------
def tune_thresholds_f1(Y_true: np.ndarray, Y_proba: np.ndarray, grid=None) -> np.ndarray:
    if grid is None: grid = np.linspace(0.05, 0.95, 19)
    C = Y_true.shape[1]; thr = np.zeros(C, dtype=np.float32)
    for c in range(C):
        y = Y_true[:, c].astype(int); p = Y_proba[:, c]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (p >= t).astype(int)
            tp = (pred & y).sum(); fp = (pred & (1-y)).sum(); fn = ((1-pred) & y).sum()
            pr = tp / max(tp+fp, 1e-9); rc = tp / max(tp+fn, 1e-9)
            f1 = 0.0 if (pr+rc)==0 else 2*pr*rc/(pr+rc)
            if f1 > best_f1: best_f1, best_t = f1, t
        thr[c] = best_t
    return thr

def per_class_metrics(Y_true: np.ndarray, Y_pred: np.ndarray):
    tp_all = fp_all = fn_all = 0; f1s = []; per = {}
    for i, lbl in enumerate(LABELS):
        y = Y_true[:, i].astype(bool); p = Y_pred[:, i].astype(bool)
        tp = int((p & y).sum()); fp = int(((1 - y) & p).sum()); fn = int((y & (1 - p)).sum())
        pr = tp / max(tp+fp, 1e-9); rc = tp / max(tp+fn, 1e-9)
        f1 = 0.0 if (pr+rc)==0 else 2*pr*rc/(pr+rc)
        per[lbl] = {"P": pr, "R": rc, "F1": f1, "TP": tp, "FP": fp, "FN": fn}
        tp_all += tp; fp_all += fp; fn_all += fn
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    micro_pr = tp_all / max(tp_all + fp_all, 1e-9)
    micro_rc = tp_all / max(tp_all + fn_all, 1e-9)
    micro_f1 = 0.0 if (micro_pr+micro_rc)==0 else 2*micro_pr*micro_rc/(micro_pr+micro_rc)
    return per, macro_f1, micro_f1

# -----------------------------
# Training
# -----------------------------
def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Load & aggregate
    df = pd.read_csv(cfg.csv_path)
    texts_df, Y_all = aggregate_articles(df, cfg.text_cols, "EventLabel", cfg.key_preference)
    texts = texts_df["text_feat"].tolist()

    # 2) Split (random, article-level; no stratify to avoid 1-sample classes issues)
    n = len(texts); idx = np.arange(n); rng = np.random.RandomState(cfg.seed)
    rng.shuffle(idx); k = int(0.8 * n)
    tr_idx, va_idx = idx[:k], idx[k:]
    texts_tr = [texts[i] for i in tr_idx]; texts_va = [texts[i] for i in va_idx]
    Y_tr, Y_va = Y_all[tr_idx], Y_all[va_idx]

    # 3) Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model, use_fast=True)
    ds_tr = TextDataset(texts_tr, Y_tr, tokenizer, cfg.max_length)
    ds_va = TextDataset(texts_va, Y_va, tokenizer, cfg.max_length)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    # 4) Model
    model = MultiLabelTransformer(cfg.pretrained_model, N_CLASSES, dropout=cfg.dropout).to(cfg.device)

    # 5) Loss & optimizer
    pos_weight = effective_pos_weight(Y_tr).to(cfg.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Scheduler (linear warmup)
    total_steps = len(dl_tr) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # 6) Train loop with early stopping by macro-F1 (@0.5)
    best_macro = -1.0; best_state = None; patience = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train(); running = 0.0
        for batch in dl_tr:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                if cfg.use_focal:
                    loss = focal_bce_with_logits(logits, labels, alpha=pos_weight, gamma=cfg.focal_gamma)
                else:
                    loss = bce(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            running += float(loss.item())

        # Validation
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dl_va:
                input_ids = batch["input_ids"].to(cfg.device)
                attention_mask = batch["attention_mask"].to(cfg.device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        Y_proba = np.concatenate(all_probs, axis=0)

        # Metrics @0.5
        Y_pred_05 = (Y_proba >= 0.5).astype(int)
        print(">>>> 05", Y_pred_05)
        _, macro_05, micro_05 = per_class_metrics(Y_va, Y_pred_05)

        print(f"Epoch {epoch:02d} | TrainLoss {running/len(dl_tr):.4f} | Val macro-F1@0.5 {macro_05:.3f} | micro-F1@0.5 {micro_05:.3f}")

        # Early stopping
        if macro_05 > best_macro:
            best_macro = macro_05
            best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.es_patience:
                print("Early stopping triggered.")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Threshold tuning on validation probs
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dl_va:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    Y_proba = np.concatenate(all_probs, axis=0)

    thr = tune_thresholds_f1(Y_va, Y_proba)
    Y_hat = (Y_proba >= thr).astype(int)
    per_t, macro_t, micro_t = per_class_metrics(Y_va, Y_hat)

    print("\nPer-class metrics (tuned thresholds):")
    for lbl in LABELS:
        m = per_t[lbl]
        print(f"{lbl:12s}  P={m['P']:.3f}  R={m['R']:.3f}  F1={m['F1']:.3f}  TP={m['TP']} FP={m['FP']} FN={m['FN']}")
    print(f"\nMacro-F1 (tuned): {macro_t:.3f} | Micro-F1 (tuned): {micro_t:.3f}")

    # 8) Save artifacts
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_state.pt"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)
    with open(os.path.join(cfg.out_dir, "thresholds.json"), "w") as f:
        json.dump({lbl: float(t) for lbl, t in zip(LABELS, thr)}, f, indent=2)
    print(f"\nSaved model, config, and thresholds to {cfg.out_dir}")
    return model, thr, (texts_va, Y_va, Y_proba)

# -----------------------------
# Inference helper
# -----------------------------
def predict_proba(model: nn.Module, tokenizer: AutoTokenizer, texts: List[str],
                  device: str, max_length: int) -> np.ndarray:
    model.eval()
    probs_all = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)

if __name__ == "__main__":
    cfg = Config(
        csv_path="../data/train_20251216161539.csv",
        # Switch to Longformer for longer articles:
        # pretrained_model="allenai/longformer-base-4096", max_length=1024 or 2048 or 4096
        pretrained_model="roberta-base",
        max_length=512,
        use_focal=False,   # set True to try focal loss
    )
    train(cfg)
