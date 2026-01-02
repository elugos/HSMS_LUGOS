
# tfidf_cosine_dedupe.py
"""
Near-duplicate document cleanup using TF-IDF + cosine similarity.

Features
- Text normalization (lowercase, punctuation removal, collapses whitespace)
- Exact duplicate removal via hashing
- TF-IDF vectorization with configurable n-grams and min_df
- Cosine similarity search using scikit-learn NearestNeighbors (sparse-friendly)
- Threshold-based deduplication: keep a representative per duplicate group
- Optional blocking by length (speeds up) and domain/metadata
- Command-line interface and importable functions

Usage (CLI)
    python tfidf_cosine_dedupe.py --input data.csv --text-col text \
        --output deduped.csv --threshold 0.92 --min-df 5 --ngram-min 1 --ngram-max 2

Usage (library)
    from tfidf_cosine_dedupe import dedupe_tfidf_cosine
    cleaned_df, report = dedupe_tfidf_cosine(df, text_col='text', threshold=0.92)

Notes
- This pipeline first removes exact duplicates by hash, then finds near duplicates using TF-IDF cosine.
- For large datasets (>100k docs), consider running with blocking or chunking, or switch to LSH/FAISS.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# =============================
# Text normalization & hashing
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep alnum + spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def hash_text(s: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(s.encode("utf-8"))
    return h.hexdigest()

# =============================
# TF-IDF + cosine dedupe
def tfidf_vectors(texts: List[str], min_df: int = 3, ngram_range: Tuple[int, int] = (1, 2), max_features: Optional[int] = None):
    vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, max_features=max_features)
    X = vec.fit_transform(texts)
    return vec, X

def build_nn_index(X, metric: str = "cosine", n_neighbors: int = 5):
    # cosine in sklearn NearestNeighbors computes pairwise distances; similarity = 1 - distance
    nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X)
    return nn

def find_near_duplicates(X, nn, threshold: float = 0.9) -> Dict[int, List[int]]:
    """Return mapping {i: [j1, j2, ...]} of near-duplicates for each i (including i itself).
       Similarity = 1 - cosine_distance.
    """
    n = X.shape[0]
    groups: Dict[int, List[int]] = {}
    distances, neighbors = nn.kneighbors(X, return_distance=True)
    for i in range(n):
        sims = 1.0 - distances[i]
        cand = [int(neighbors[i][k]) for k in range(neighbors.shape[1]) if sims[k] >= threshold]
        # Ensure self included
        if i not in cand:
            cand.append(i)
        groups[i] = sorted(set(cand))
    return groups

def merge_groups(groups: Dict[int, List[int]]) -> List[List[int]]:
    """Merge overlapping groups into disjoint components (Union-Find via DFS)."""
    adj: Dict[int, set] = {i: set() for i in groups.keys()}
    for i, js in groups.items():
        for j in js:
            adj[i].add(j)
            adj.setdefault(j, set()).add(i)
    visited = set()
    comps: List[List[int]] = []
    for i in adj.keys():
        if i in visited:
            continue
        stack = [i]
        comp = []
        visited.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps

def choose_representatives(df: pd.DataFrame, comps: List[List[int]], prefer_longer: bool = True, text_col: str = "text") -> List[int]:
    reps = []
    lengths = df[text_col].fillna("").str.len().to_numpy()
    for comp in comps:
        if prefer_longer:
            rep = max(comp, key=lambda idx: lengths[idx])
        else:
            rep = min(comp)  # fallback deterministic choice
        reps.append(rep)
    return reps

def dedupe_tfidf_cosine(
    df: pd.DataFrame,
    text_col: str,
    threshold: float = 0.90,
    min_df: int = 3,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: Optional[int] = None,
    prefer_longer: bool = True,
    block_by_length: bool = True,
    length_tolerance: float = 0.15,
    return_groups: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Run deduplication and return (deduped_df, report).

    Strategy:
      1) Normalize + hash -> drop exact duplicates.
      2) Optional length blocking: compare only docs whose lengths are within +/- tol.
      3) TF-IDF vectors + cosine NN -> build neighbor groups above threshold.
      4) Merge overlapping groups and keep a representative per group.

    Report contains statistics and (optionally) group assignments.
    """
    work = df.reset_index(drop=True).copy()
    # Step 1: normalize + hash
    work["__norm_text"] = work[text_col].apply(normalize_text)
    work["__hash"] = work["__norm_text"].apply(hash_text)

    # Drop exact duplicates by hash, keep first
    before = len(work)
    work = work.drop_duplicates(subset=["__hash"], keep="first").reset_index(drop=True)
    after_exact = len(work)

    # Optional length blocking
    if block_by_length:
        lengths = work["__norm_text"].str.len().to_numpy()
        work["__len"] = lengths
        # Create bins so that docs only compare with similar lengths
        # Bin width proportional to length tolerance
        bins = np.maximum(1, (lengths * length_tolerance).astype(int))
        # Simple bucketing: rounded length to nearest bin multiple
        work["__bucket"] = (lengths // np.maximum(1, bins)).astype(int)
    else:
        work["__bucket"] = 0

    dedup_indices_keep = []
    all_groups: List[List[int]] = []
    # Process bucket by bucket
    for bval, sub in work.groupby("__bucket"):
        if len(sub) == 1:
            dedup_indices_keep.append(sub.index[0])
            continue
        texts = sub["__norm_text"].tolist()
        vec, X = tfidf_vectors(texts, min_df=min_df, ngram_range=(ngram_min, ngram_max), max_features=max_features)
        nn = build_nn_index(X, metric="cosine", n_neighbors=min(5, len(sub)))
        groups_local = find_near_duplicates(X, nn, threshold=threshold)
        comps = merge_groups(groups_local)
        reps_local = choose_representatives(sub, comps, prefer_longer=prefer_longer, text_col="__norm_text")
        # Map local positions to original indices
        orig_idx = sub.index.to_numpy()
        keep_idx = [int(orig_idx[r]) for r in reps_local]
        dedup_indices_keep.extend(keep_idx)
        # Store groups with original indices
        all_groups.extend([[int(orig_idx[j]) for j in comp] for comp in comps])

    dedup_df = work.loc[sorted(set(dedup_indices_keep))].drop(columns=["__norm_text", "__hash", "__len", "__bucket"], errors="ignore")
    removed = before - len(dedup_df)

    report = {
        "n_input": before,
        "n_after_exact": after_exact,
        "n_output": len(dedup_df),
        "n_removed": removed,
        "threshold": threshold,
        "min_df": min_df,
        "ngram_range": (ngram_min, ngram_max),
    }
    if return_groups:
        report["groups"] = all_groups

    return dedup_df.reset_index(drop=True), report

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Near-duplicate cleanup with TF-IDF + cosine similarity")
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--text-col", required=True, help="Column name with text")
    ap.add_argument("--output", required=True, help="Output CSV file (deduped)")
    ap.add_argument("--threshold", type=float, default=0.92, help="Cosine similarity threshold (0-1)")
    ap.add_argument("--min-df", type=int, default=3, help="TF-IDF min_df")
    ap.add_argument("--ngram-min", type=int, default=1, help="TF-IDF ngram min")
    ap.add_argument("--ngram-max", type=int, default=2, help="TF-IDF ngram max")
    ap.add_argument("--max-features", type=int, default=None, help="TF-IDF max_features")
    ap.add_argument("--no-length-block", action="store_true", help="Disable length-based blocking")
    ap.add_argument("--prefer-short", action="store_true", help="Prefer shorter document as representative")
    ap.add_argument("--return-groups", action="store_true", help="Include groups in report JSON next to output")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    prefer_longer = not args.prefer_short
    block_by_length = not args.no_length_block

    dedup_df, report = dedupe_tfidf_cosine(
        df=df,
        text_col=args.text_col,
        threshold=args.threshold,
        min_df=args.min_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
        prefer_longer=prefer_longer,
        block_by_length=block_by_length,
        return_groups=args.return_groups,
    )
    dedup_df.to_csv(args.output, index=False)
    print("Saved deduped CSV:", args.output)
    print("Report:", report)

if __name__ == "__main__":
    main()
