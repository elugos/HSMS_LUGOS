#!/usr/bin/env python3
"""
bulk_spectral_embed.py
-----------------------

Bulk process CSV files that contain a column `embedding_json`. For each CSV:

1. Read the file.
2. Parse embeddings into a dense matrix.
3. Compute cosine pairwise distances.
4. Convert distance matrix to adjacency matrix.
5. Run spectral embedding.
6. Save output CSV with embedding coordinates.

Usage:
    python bulk_spectral_embed.py \
        --input-dir /path/to/csvs \
        --output-dir /path/to/out \
        --components 40
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.manifold import SpectralEmbedding
from scipy import sparse


# --------------------------------------------------
# Utility: parse JSON-like embedding into np.array
# --------------------------------------------------
def parse_embedding_column(df, col="embedding_json"):
    """Convert embedding_json column into a 2D float32 numpy array."""
    s = df[col].astype(str).str.strip()
    s = s.str.lstrip('[').str.rstrip(']')
    s = s.str.split(',')
    vecs = [
        np.array(list(map(float, parts)), dtype=np.float32)
        for parts in s
    ]
    return np.vstack(vecs)  # shape: (N, D)


# --------------------------------------------------
# Pairwise cosine distances → adjacency
# --------------------------------------------------
def compute_adjacency(X):
    """
    Compute cosine distance matrix and convert to adjacency:
        adjacency = 1 - cosine_distance
    """
    print("   • Computing pairwise cosine distances…")
    D = pairwise_distances(X, metric="cosine")

    print("   • Converting distance matrix → adjacency…")
    A = 1.0 - D
    np.fill_diagonal(A, 0.0)
    return A


# --------------------------------------------------
# Spectral embedding wrapper
# --------------------------------------------------
def spectral_embed_from_adjacency(
    A,
    node_ids=None,
    n_components=2,
    make_symmetric=True,
    zero_self_loops=True,
    use_sparse=True,
    random_state=42
):
    """
    Simplified version for bulk processing.
    """
    A = np.asarray(A)
    n = A.shape[0]

    if node_ids is None:
        node_ids = np.arange(n)

    if make_symmetric:
        A = 0.5 * (A + A.T)
    if zero_self_loops:
        np.fill_diagonal(A, 0.0)

    degrees = A.sum(axis=1)
    valid_mask = degrees > 0

    if valid_mask.any():
        A_valid = A[np.ix_(valid_mask, valid_mask)]
        if use_sparse:
            A_valid = sparse.csr_matrix(A_valid)

        emb = SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",
            random_state=random_state,
            n_jobs=1
        )
        coords_valid = emb.fit_transform(A_valid)
    else:
        coords_valid = np.empty((0, n_components))

    # Build column names
    colnames = ["x", "y"] + [f"x{i}" for i in range(2, n_components)]

    # Insert NaN rows for isolates
    out = np.full((n, n_components), np.nan)
    out[valid_mask] = coords_valid

    df = pd.DataFrame(out, columns=colnames)
    df.insert(0, "node_id", node_ids)

    return df


# --------------------------------------------------
# Main bulk processor
# --------------------------------------------------
def process_one_csv(path_in: Path, outdir: Path, n_components: int):
    print(f"\n=== Processing {path_in.name} ===")

    df = pd.read_csv(path_in)
    print(f"   • Loaded {len(df):,} rows.")

    # 1. Parse embeddings → matrix X
    print("   • Parsing embedding_json → matrix…")
    X = parse_embedding_column(df)

    # 2. Pairwise distances → adjacency
    A = compute_adjacency(X)

    # 3. Spectral embedding
    print("   • Running spectral embedding…")
    df_emb = spectral_embed_from_adjacency(
        A,
        node_ids=df.index.values,
        n_components=n_components,
        make_symmetric=True,
        zero_self_loops=True,
        use_sparse=True,
        random_state=42
    )

    # 4. Save output
    outpath = outdir / f"{path_in.stem}_spectral.csv"
    df_emb.to_csv(outpath, index=False)
    print(f"   • Saved → {outpath}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--components", type=int, default=40)
    args = ap.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*.csv"))
    if not files:
        print("No CSV files found in input directory.")
        return

    print(f"Found {len(files)} CSV files.\n")

    for f in files:
        process_one_csv(f, outdir, args.components)


if __name__ == "__main__":
    main()
