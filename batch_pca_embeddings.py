#!/usr/bin/env python3
"""
pca_pad_dense.py
One global PCA over CSVs with dense embeddings of mixed length.

Usage examples:
  # Recommended: pool token embeddings (nested lists) to fixed-D, then PCA
  python3 pca_pad_dense.py --input /path/in --output /path/out --n-components 0.95 --strategy pool

  # Pad 1D vectors to modal length, mean-impute padded positions, then PCA
  python3 pca_pad_dense.py --input /path/in --output /path/out --n-components 50 \
      --strategy pad --target-length mode

  # Truncate to min length, no padding needed
  python3 pca_pad_dense.py --input /path/in --output /path/out --n-components 100 \
      --strategy truncate --target-length min
"""

import os, glob, argparse, ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA

def read_vals(ser: pd.Series):
    """Return Python objects from a column that may already be lists;
    if strings look like lists, literal_eval; leave lists/arrays as-is."""
    out = []
    for x in ser.values:
        if isinstance(x, str) and x.startswith('['):
            x = ast.literal_eval(x)
        out.append(x)
    return out

def is_nested(v):
    """True if v is a sequence of sequences (token embeddings)."""
    return hasattr(v, '__len__') and len(v) > 0 and hasattr(v[0], '__len__')

def lengths_1d(vals):
    """Return list of lengths for 1D vectors; raise if nested detected."""
    lens = []
    for v in vals:
        if is_nested(v):
            raise ValueError("Detected nested/token embeddings but strategy is for 1D vectors. Use --strategy pool.")
        if not hasattr(v, '__len__'):
            raise TypeError("Non-list embedding encountered.")
        lens.append(len(v))
    return lens

def mean_pool_nested(vals):
    """Mean-pool (T x D) arrays -> (D,)"""
    pooled = []
    for v in vals:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError("Expected nested list/2D array per row (tokens x dim).")
        pooled.append(a.mean(axis=0))
    return pooled

def choose_target_length(lens, target):
    if isinstance(target, str):
        t = target.lower()
        if t == 'min': return int(min(lens))
        if t == 'max': return int(max(lens))
        if t == 'mode': return Counter(lens).most_common(1)[0][0]
        raise ValueError("target-length must be min|max|mode|<int>")
    return int(target)

def pad_right(vec, L):
    a = np.asarray(vec, dtype=np.float32)
    if a.shape[0] >= L: return a[:L]
    z = np.zeros(L - a.shape[0], dtype=np.float32)
    return np.concatenate([a, z], axis=0)

def build_matrix_with_mask(vals, L, mode):
    """Pad/Truncate to L and also return a mask of real positions (True=real, False=pad).
       mode: 'pad' or 'truncate' (both produce length L)."""
    X = np.zeros((len(vals), L), dtype=np.float32)
    M = np.zeros((len(vals), L), dtype=np.bool_)
    for i, v in enumerate(vals):
        a = np.asarray(v, dtype=np.float32)
        if mode == 'truncate':
            take = min(L, a.shape[0])
            X[i, :take] = a[:take]
            M[i, :take] = True
        else:  # pad
            if a.shape[0] >= L:
                X[i, :] = a[:L]
                M[i, :] = True
            else:
                X[i, :a.shape[0]] = a
                M[i, :a.shape[0]] = True
    return X, M

def mean_impute_padded(X, M):
    """For columns, compute mean over real entries only; fill padded positions with that mean."""
    # Avoid division by zero in corner cases
    counts = M.sum(axis=0)
    # For features with zero real entries (unlikely), leave zeros.
    safe = counts > 0
    means = np.zeros(X.shape[1], dtype=np.float32)
    means[safe] = (X[:, safe] * M[:, safe]).sum(axis=0) / counts[safe]
    # Fill only where M is False (padded positions)
    X_filled = X.copy()
    pad_idx = ~M
    X_filled[pad_idx] = np.take(means, np.nonzero(pad_idx)[1])
    return X_filled

def fit_global_pca(X_all, n_components, svd_solver, seed):
    ncomp = int(n_components) if n_components > 1 else float(n_components)
    pca = PCA(n_components=float(args.n_components), svd_solver="full", random_state=args.seed)
    pca.fit(X_all)
    return pca

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--col", default="embedding_json")
    ap.add_argument("--strategy", choices=["pool", "pad", "truncate"], default="pool",
                    help="pool=mean-pool nested token embeddings; pad/truncate for 1D vectors.")
    ap.add_argument("--target-length", default="mode",
                    help="For pad/truncate: min|max|mode|<int> (default: mode)")
    ap.add_argument("--n-components", type=float, required=True,
                    help="Int (e.g., 50) or variance fraction (0,1], e.g., 0.95")
    ap.add_argument("--svd-solver", default="randomized", choices=["auto","full","arpack","randomized"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {args.input}")

    # Load all files
    dfs, all_vals, per_file_vals = [], [], []
    for fp in files:
        df = pd.read_csv(fp)
        if args.col not in df.columns:
            raise KeyError(f"Column '{args.col}' not found in {fp}")
        vals = read_vals(df[args.col])
        dfs.append(df)
        per_file_vals.append(vals)
        all_vals.extend(vals)

    # Build a single matrix for PCA (global), depending on strategy
    if args.strategy == "pool":
        # Expect nested token embeddings -> mean-pool to fixed D
        pooled_all, pooled_per_file = [], []
        for vals in per_file_vals:
            pooled = mean_pool_nested(vals)  # list of (D,)
            pooled_per_file.append(pooled)
            pooled_all.extend(pooled)
        X_all = np.vstack([np.asarray(v, dtype=np.float32) for v in pooled_all])

    else:
        # 1D vectors with varying lengths -> align lengths
        lens = lengths_1d(all_vals)
        if len(set(lens)) == 1:
            L = lens[0]
        else:
            L = choose_target_length(lens, args.target_length)
        # Build per-file matrices with mask, then mean-impute pad positions
        adjusted_all = []
        adjusted_per_file = []
        for vals in per_file_vals:
            X, M = build_matrix_with_mask(vals, L, mode=args.strategy)
            X_filled = mean_impute_padded(X, M)
            adjusted_per_file.append(X_filled)
            adjusted_all.append(X_filled)
        X_all = np.vstack(adjusted_all)

    # Fit PCA globally
    pca = fit_global_pca(X_all, args.n_components, args.svd_solver, args.seed)

    # Diagnostics
    print(f"Files: {len(files)}")
    print(f"Total rows: {X_all.shape[0]}")
    print(f"Input dim: {X_all.shape[1]}")
    print(f"PCA components: {pca.n_components_}")
    if hasattr(pca, "explained_variance_ratio_"):
        print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.6f}")

    # Transform and save
    for fp, df, vals in zip(files, dfs, per_file_vals):
        if args.strategy == "pool":
            pooled = mean_pool_nested(vals)
            X = np.vstack([np.asarray(v, dtype=np.float32) for v in pooled])
        else:
            X, M = build_matrix_with_mask(vals, X_all.shape[1], mode=args.strategy)
            X = mean_impute_padded(X, M)
        Xr = pca.transform(X)
        df["embedding_pca"] = [row.tolist() for row in Xr]
        out_fp = os.path.join(args.output, os.path.basename(fp))
        df.to_csv(out_fp, index=False)
        print(f"Saved: {out_fp}")

if __name__ == "__main__":
    main()
