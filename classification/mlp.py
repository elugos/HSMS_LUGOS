
# train_sbert_mlp.py
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------
# Config
# -----------------------------
SEED = 1
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Keep label order consistent with your dataset
LABELS = [
    "CONSULT","ASSAULT","DISAPPROVE","SUPPORT","COERCE","AGREE","AID","REJECT",
    "CONCEDE","COOPERATE","RETREAT","THREATHEN","DEMAND","PROTEST","SANCTION","MOBILIZE"
]
label2id = {l:i for i,l in enumerate(LABELS)}
N_CLASSES = len(LABELS)

@dataclass
class Config:
    csv_path: str = "train_sample.csv"
    # choose embeddings column present in your CSV:
    emb_col: str = "embedding_json_title"   # or "first_para_emb"
    label_col: str = "EventLabel"
    key_preference: Tuple[str, ...] = ("SOURCEURL", "GlobalEventID")  # article key
    batch_size: int = 64
    lr: float = 2e-3
    epochs: int = 10
    hidden: int = 256
    dropout: float = 0.1
    use_focal: bool = False      # set True to use focal BCE
    focal_gamma: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "outputs_sbert_mlp"

# -----------------------------
# Utils: thresholds & metrics
# -----------------------------
def tune_thresholds_f1(Y_true: np.ndarray, Y_proba: np.ndarray, grid=None) -> np.ndarray:
    """Return per-class thresholds that maximize F1 on validation."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    C = Y_true.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    for c in range(C):
        y = Y_true[:, c].astype(np.int32)
        p = Y_proba[:, c]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (p >= t).astype(np.int32)
            tp = (pred & y).sum()
            fp = (pred & (1-y)).sum()
            fn = ((1-pred) & y).sum()
            precision = tp / max(tp+fp, 1e-9)
            recall    = tp / max(tp+fn, 1e-9)
            f1 = 0.0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr

def per_class_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[dict, float, float]:
    """Return per-class P/R/F1 and macro/micro F1."""
    C = Y_true.shape[1]
    per = {}
    tp_all = fp_all = fn_all = 0
    f1s = []
    for i, lbl in enumerate(LABELS):
        y = Y_true[:, i]; p = Y_pred[:, i]
        tp = int((p & y).sum())
        fp = int(((1 - y) & p).sum())
        fn = int((y & (1 - p)).sum())
        precision = tp / max(tp+fp, 1e-9)
        recall    = tp / max(tp+fn, 1e-9)
        f1 = 0.0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)
        per[lbl] = {"P": precision, "R": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn}
        tp_all += tp; fp_all += fp; fn_all += fn
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    micro_precision = tp_all / max(tp_all + fp_all, 1e-9)
    micro_recall    = tp_all / max(tp_all + fn_all, 1e-9)
    micro_f1 = 0.0 if (micro_precision+micro_recall)==0 else 2*micro_precision*micro_recall/(micro_precision+micro_recall)
    return per, macro_f1, micro_f1

# -----------------------------
# Data aggregation & parsing
# -----------------------------
def _parse_emb(s: str) -> Optional[np.ndarray]:
    try:
        arr = np.array(json.loads(s), dtype=np.float32)
        # guard against empty arrays
        if arr.ndim == 1 and arr.size > 0:
            return arr
        return None
    except Exception:
        return None

def aggregate_articles_with_embeddings(
    df: pd.DataFrame,
    emb_col: str,
    label_col: str,
    key_preference: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group by article key, collect multi-hot labels, and pick/average embeddings per article.
    Returns (X_emb[N,d], Y[N,C]).
    """
    df[label_col] = df[label_col].astype(str).str.upper()
    df = df[df[label_col].isin(LABELS)].copy()

    # Choose article key
    key = None
    for k in key_preference:
        if k in df.columns and not df[k].isna().all():
            key = k; break
    if key is None:
        # fallback: hash over title+first_para to prevent leakage
        key = "__article_key__"
        df[key] = (df.get("title","").fillna("").astype(str) + "||" +
                   df.get("first_para","").fillna("").astype(str))

    # Parse embeddings and drop rows without embeddings
    df["_emb"] = df[emb_col].apply(lambda s: _parse_emb(s) if isinstance(s, str) else None)
    df = df[~df["_emb"].isna()].copy()

    # Aggregate: labels set + embedding (average across duplicates)
    grouped = []
    for gkey, g in df.groupby(key):
        labels = list(set(g[label_col].tolist()))
        # average embeddings across rows for this article
        embs = [e for e in g["_emb"].tolist() if e is not None]
        if len(embs) == 0:
            continue
        # confirm equal dim; otherwise pick first
        dim = embs[0].shape[0]
        embs = [e for e in embs if e.shape[0] == dim]
        emb_avg = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)

        y = np.zeros(N_CLASSES, dtype=np.float32)
        for lbl in labels:
            y[label2id[lbl]] = 1.0
        grouped.append((emb_avg, y))

    if len(grouped) == 0:
        raise ValueError("No valid articles after aggregation. Check emb_col and label mapping.")

    X = np.stack([t[0] for t in grouped], axis=0)
    Y = np.stack([t[1] for t in grouped], axis=0).astype(int)
    return X, Y

class EmbDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.dim = self.X.shape[1]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# -----------------------------
# Model: small MLP head
# -----------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)  # logits

# -----------------------------
# Losses: weighted BCE + focal
# -----------------------------
def effective_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute per-class pos_weight from training labels.
    Inverse-sqrt weighting: pos_weight ~ max/sqrt(count) (clamped to >=1).
    """
    counts = y_train.sum(axis=0)  # per class positive counts
    mx = float(np.max(counts))
    w = mx / np.sqrt(np.clip(counts, 1.0, None))
    w = np.clip(w, 1.0, None).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32)

def focal_bce_with_logits(logits, targets, alpha=None, gamma: float = 2.0, reduction="mean"):
    """
    Multi-label focal BCE. alpha can be a tensor of per-class weights.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=alpha, reduction="none"
    )
    pt = torch.exp(-bce)
    loss = ((1 - pt) ** gamma) * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

# -----------------------------
# Sampler: oversample rare-label samples
# -----------------------------
def make_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights = sum_c (y_ic * class_weight_c),
    where class_weight_c ~ 1/sqrt(freq_c).
    This oversamples samples that carry rare labels.
    """
    counts = y_train.sum(axis=0)
    cw = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    s_w = (y_train * cw).sum(axis=1)
    # avoid zeros
    s_w = np.clip(s_w, 1e-3, None).astype(np.float32)
    return s_w

# -----------------------------
# Training / Evaluation
# -----------------------------
def train(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    df = pd.read_csv(cfg.csv_path)

    # Aggregate to article-level multi-label
    X_all, Y_all = aggregate_articles_with_embeddings(
        df, cfg.emb_col, cfg.label_col, cfg.key_preference
    )

    # Split (group-level random, no stratify to avoid 1-sample classes issue)
    n = X_all.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    k = int(0.8 * n)
    tr_idx, va_idx = idx[:k], idx[k:]
    X_tr, Y_tr = X_all[tr_idx], Y_all[tr_idx]
    X_va, Y_va = X_all[va_idx], Y_all[va_idx]

    # Datasets
    ds_tr = EmbDataset(X_tr, Y_tr)
    ds_va = EmbDataset(X_va, Y_va)

    # Model
    in_dim = ds_tr.dim
    model = MLPHead(in_dim, N_CLASSES, hidden=cfg.hidden, p=cfg.dropout).to(cfg.device)

    # Loss: weighted BCE or focal BCE
    pos_weight = effective_pos_weight(Y_tr).to(cfg.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Sampler (optional but helpful for imbalance)
    sample_weights = make_sample_weights(Y_tr)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    # Train loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for Xb, Yb in dl_tr:
            Xb = Xb.to(cfg.device); Yb = Yb.to(cfg.device)
            opt.zero_grad()
            logits = model(Xb)
            if cfg.use_focal:
                loss = focal_bce_with_logits(logits, Yb, alpha=pos_weight, gamma=cfg.focal_gamma)
            else:
                loss = bce(logits, Yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        avg_tr = total / max(1, len(dl_tr))

        # Quick validation metrics @0.5
        model.eval()
        with torch.no_grad():
            all_probs = []
            for Xb, _ in dl_va:
                Xb = Xb.to(cfg.device)
                probs = torch.sigmoid(model(Xb)).cpu().numpy()
                all_probs.append(probs)
            Y_proba = np.concatenate(all_probs, axis=0)
            Y_pred_05 = (Y_proba >= 0.5).astype(np.int32)
            per_05, macro_05, micro_05 = per_class_metrics(Y_va, Y_pred_05)

        print(f"Epoch {epoch:02d} | TrainLoss {avg_tr:.4f} | Val macro-F1@0.5 {macro_05:.3f} | micro-F1@0.5 {micro_05:.3f}")

    # Threshold tuning on validation set (F1-max per class)
    thr = tune_thresholds_f1(Y_va, Y_proba)
    Y_hat = (Y_proba >= thr).astype(np.int32)
    per_t, macro_t, micro_t = per_class_metrics(Y_va, Y_hat)

    print("\nPer-class metrics (tuned thresholds):")
    for lbl in LABELS:
        m = per_t[lbl]
        print(f"{lbl:12s}  P={m['P']:.3f}  R={m['R']:.3f}  F1={m['F1']:.3f}  TP={m['TP']} FP={m['FP']} FN={m['FN']}")
    print(f"\nMacro-F1 (tuned): {macro_t:.3f} | Micro-F1 (tuned): {micro_t:.3f}")

    # Save artifacts
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "hidden": cfg.hidden}, os.path.join(cfg.out_dir, "mlp_head.pt"))
    with open(os.path.join(cfg.out_dir, "thresholds.json"), "w") as f:
        json.dump({lbl: float(t) for lbl, t in zip(LABELS, thr)}, f, indent=2)
    print(f"\nSaved model and thresholds to: {cfg.out_dir}")

    return model, thr, (X_va, Y_va, Y_proba)

# -----------------------------
# Inference helper
# -----------------------------
def predict_probs(model: nn.Module, X_emb: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X_emb.astype(np.float32)).to(device)
        probs = torch.sigmoid(model(X)).cpu().numpy()
    return probs

if __name__ == "__main__":
    cfg = Config(
        csv_path="../data/train_20251216161539.csv",
        emb_col="embedding_json_title",  # switch to "first_para_emb" if preferred
        use_focal=False,                 # set to True to try focal loss
        epochs=50
    )

    train(cfg)
