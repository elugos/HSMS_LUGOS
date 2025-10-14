#!/usr/bin/env python3
"""
Cluster co-occurrence pairs from a CSV.

Usage:
    python cluster_cooccurrences.py \
        --csv input.csv \
        --text-col sbert_text \
        --cluster-col cluster \
        --min-count 1 \
        --output cooccurrences.csv \
        --exclude-noise \
        --no-dedup
"""

import argparse
from collections import Counter
from itertools import combinations
import pandas as pd

def cluster_cooccurrences(
    df: pd.DataFrame,
    text_col: str = "sbert_text",
    cluster_col: str = "cluster",
    dedup_within_cluster: bool = True,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Count how often two texts co-appear in the same cluster.

    Args:
        df: DataFrame with at least [text_col, cluster_col].
        text_col: Column containing the text (e.g., 'sbert_text').
        cluster_col: Column containing cluster labels (e.g., 'cluster').
        dedup_within_cluster: If True, treat multiple occurrences of the SAME text
            in a single cluster as one (prevents inflated counts).
        min_count: Filter pairs with total count < min_count.

    Returns:
        DataFrame with columns: ['cluster', 'text_a', 'text_b', 'count'].
    """
    pair_rows = []

    # Group by cluster and count unordered pairs within each cluster
    for clust, g in df.groupby(cluster_col, dropna=False):
        texts = g[text_col]
        # Ensure strings; drop NaN
        texts = pd.Index(texts.astype(str)).dropna()

        if dedup_within_cluster:
            texts = texts.unique()

        # Build unordered, lexicographically sorted pairs
        pairs = (tuple(sorted(p)) for p in combinations(texts, 2))
        c = Counter(pairs)

        if not c:
            continue

        for (a, b), cnt in c.items():
            if cnt >= min_count:
                pair_rows.append((clust, a, b, cnt))

    out = pd.DataFrame(pair_rows, columns=["cluster", "text_a", "text_b", "count"])
    if out.empty:
        return out
    return out.sort_values(["cluster", "count"], ascending=[True, False]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute within-cluster text co-occurrences from a CSV."
    )
    parser.add_argument("--csv", required=True, help="Input CSV with text & cluster columns.")
    parser.add_argument("--text-col", default="sbert_text", help="Text column name.")
    parser.add_argument("--cluster-col", default="cluster", help="Cluster label column name.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum pair count to keep.")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable de-duplication within a cluster (count repeat texts multiple times).",
    )
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="Exclude rows where cluster == -1 (common noise label).",
    )
    parser.add_argument("--output", default="cooccurrences.csv", help="Output CSV path.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)

    # Basic checks
    for col in (args.text_col, args.cluster_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    # Optionally drop noise cluster (-1)
    if args.exclude_noise:
        df = df[df[args.cluster_col] != -1]

    # Compute co-occurrences
    out = cluster_cooccurrences(
        df,
        text_col=args.text_col,
        cluster_col=args.cluster_col,
        dedup_within_cluster=not args.no_dedup,
        min_count=args.min_count,
    )

    # Save and print a quick summary
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} co-occurrence rows to {args.output}")
    if not out.empty:
        print("Top 5 rows:")
        print(out.head().to_string(index=False))
    else:
        print("No co-occurrences met the criteria.")

if __name__ == "__main__":
    main()
