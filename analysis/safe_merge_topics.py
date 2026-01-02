
# safe_seed_merge.py
"""
Safe merging of BERTopic topics using seed hits, with robust helpers
and guards to avoid index errors and stale IDs.

This module provides:
  - Seed normalization and document-level seed-hit computation.
  - Topic-level seed vectors (IDF-weighted) and similarity-based merge proposals.
  - Pair sanitization against current model state.
  - Connected-components merging (merge many topics into a single representative).
  - Optional embedding gate using representative-doc centroids.
  - One-shot and multi-round merge runners.
  - Convenience helper to merge seed-rich topics into an anchor topic.

Usage (library):
    from safe_seed_merge import (
        normalize_seed_words,
        attach_doc_seed_hits,
        build_topic_seed_df,
        run_seed_merge_rounds,
        build_anchor_pairs_from_seed_df,
        merge_by_components,
    )

    # Pre-conditions: you have df, texts, embeddings, topic_model, and seed list
    seed_words = normalize_seed_words(cfg["keywords"])  # or provide your list
    df = attach_doc_seed_hits(df, texts, seed_words)    # adds per-document seed columns
    topic_seed_df = build_topic_seed_df(df)             # per-topic seed stats

    # Example A: Fully automatic multi-round seed-space merging
    topic_info_after, new_topics = run_seed_merge_rounds(
        df=df,
        topic_model=topic_model,
        texts=texts,
        embeddings=embeddings,
        seed_words=seed_words,
        topic_seed_df=topic_seed_df,
        rounds=2,
        seed_sim_threshold=0.60,
        embedder=embedder,                    # optional; for embedding gate
        require_embedding_sim=True,
        embed_sim_threshold=0.65,
        min_shared_seeds=2,
        max_merge_ratio=3.0,
        verbose=True,
    )
    df["topic"] = new_topics

    # Example B: Anchor-based merges (merge seed-rich topics into largest seed-hit topic)
    anchor = int(topic_seed_df.loc[topic_seed_df['prop_docs_with_seed_hit'].idxmax(), 'Topic'])
    candidate_pairs = build_anchor_pairs_from_seed_df(topic_seed_df, anchor, prop_threshold=0.40)
    topic_info_after = merge_by_components(topic_model, texts, candidate_pairs, embeddings=embeddings, verbose=True)
    # Refresh assignments
    new_topics, _ = topic_model.transform(texts, embeddings)
    df["topic"] = new_topics

Notes:
  - Pairs for merge_topics follow BERTopic's convention: (topic_to, topic_from).
  - Outliers (-1) are *not* merged here; reassign outlier documents separately.
  - After merges, we call topic_model.transform(...) to refresh assignments.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import re

# -----------------------------
# Basic utilities
# -----------------------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def normalize_seed_words(seed_words: List[str]) -> List[str]:
    """Normalize seeds to lowercase alpha, deduplicate and sort.
    This should match the cleaned text representation you use for BERTopic.
    """
    out = []
    for w in seed_words:
        w2 = re.sub(r"[^a-z\s]", "", str(w).lower()).strip()
        if w2:
            out.append(w2)
    return sorted(set(out))


def tokenize_cleaned(text: str) -> List[str]:
    """Tokenize a cleaned text (already lowercase, non-alpha stripped) by whitespace."""
    return text.split() if isinstance(text, str) else []

# -----------------------------
# Document-level seed hits
# -----------------------------
def compute_doc_seed_hits(texts: List[str], seed_words: List[str]) -> Tuple[List[int], List[int], List[Dict[str, int]]]:
    """Compute per-document seed hit counts.

    Returns:
        counts: total matches per doc
        uniques: number of unique seed words matched per doc
        details: dict {seed: frequency} per doc
    """
    seed_set = set(seed_words)
    counts, uniques, details = [], [], []
    for s in texts:
        toks = tokenize_cleaned(s)
        hits = [t for t in toks if t in seed_set]
        cnt = Counter(hits)
        counts.append(sum(cnt.values()))
        uniques.append(len(cnt.keys()))
        details.append(dict(cnt))
    return counts, uniques, details


def attach_doc_seed_hits(df: pd.DataFrame, texts: List[str], seed_words: List[str],
                         count_col: str = "doc_seed_hits_count",
                         unique_col: str = "doc_seed_hits_unique",
                         detail_col: str = "doc_seed_hits_detail") -> pd.DataFrame:
    """Attach doc-level seed-hit metrics to df in-place and return df."""
    counts, uniques, details = compute_doc_seed_hits(texts, seed_words)
    df[count_col] = counts
    df[unique_col] = uniques
    df[detail_col] = details
    return df

# -----------------------------
# Topic-level seed vectors & stats
# -----------------------------
def compute_topic_seed_vectors(
    df: pd.DataFrame,
    seed_words: List[str],
    topic_col: str = "topic",
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], np.ndarray]:
    """Build IDF-weighted topic–seed vectors.

    For each topic, aggregate seed frequencies across its documents, apply
    topic-IDF weighting to downweight ubiquitous seeds, normalize, and return.

    Returns:
        topic_vecs: dict {topic_id: (len(seeds),) np.ndarray}
        topic_sizes: dict {topic_id: n_docs}
        idf: (len(seeds),) np.ndarray
    """
    seed_index = {w: i for i, w in enumerate(seed_words)}
    topic_ids = sorted(t for t in set(df[topic_col]) if t != -1)

    # Per-topic raw seed counts
    raw_counts = {t: np.zeros(len(seed_words), dtype=np.float32) for t in topic_ids}
    topic_sizes = {t: int((df[topic_col] == t).sum()) for t in topic_ids}

    # Aggregate counts from detail dicts
    detail_col = "doc_seed_hits_detail"
    if detail_col not in df.columns:
        raise ValueError(f"Expected column '{detail_col}' with per-doc seed hit dicts.")

    for t in topic_ids:
        g = df[df[topic_col] == t][detail_col]
        agg = Counter()
        for d in g:
            if isinstance(d, dict) and d:
                agg.update(d)
        vec = raw_counts[t]
        for w, c in agg.items():
            idx = seed_index.get(w)
            if idx is not None:
                vec[idx] += float(c)

    # Topic presence per seed (for IDF)
    presence = np.zeros(len(seed_words), dtype=np.int32)
    for j in range(len(seed_words)):
        for t in topic_ids:
            if raw_counts[t][j] > 0:
                presence[j] += 1
    T = max(1, len(topic_ids))
    idf = np.log((T + 1) / (1 + presence)) + 1.0

    # Apply IDF, normalize, and optionally size-normalize
    topic_vecs = {}
    for t in topic_ids:
        v = raw_counts[t] * idf
        if topic_sizes[t] > 0:
            v = v / topic_sizes[t]
        n = np.linalg.norm(v)
        topic_vecs[t] = v if n == 0 else (v / n)

    return topic_vecs, topic_sizes, idf


def build_topic_seed_df(df: pd.DataFrame, topic_col: str = "topic",
                        count_col: str = "doc_seed_hits_count") -> pd.DataFrame:
    """Summarize per-topic seed incidence and totals.

    Returns a DataFrame with columns:
        Topic, n_docs_in_topic, n_docs_with_seed_hit, prop_docs_with_seed_hit, total_seed_hits
    """
    rows = []
    grouped = df[df[topic_col] != -1].groupby(topic_col)
    for t, g in grouped:
        n_docs = len(g)
        n_docs_with_seed = int((g[count_col] > 0).sum())
        prop = (n_docs_with_seed / n_docs) if n_docs else np.nan
        total = int(g[count_col].sum())
        rows.append({
            "Topic": int(t),
            "n_docs_in_topic": n_docs,
            "n_docs_with_seed_hit": n_docs_with_seed,
            "prop_docs_with_seed_hit": prop,
            "total_seed_hits": total,
        })
    return pd.DataFrame(rows).sort_values("n_docs_in_topic", ascending=False).reset_index(drop=True)

# -----------------------------
# Representative embeddings (optional gate)
# -----------------------------
def topic_rep_embedding(topic_model, embedder, df: pd.DataFrame,
                        embeddings: np.ndarray, topic_id: int, k: int = 10) -> Optional[np.ndarray]:
    """Compute a representative vector for a topic using representative docs or fallback centroid."""
    if embedder is not None:
        reps = topic_model.get_representative_docs(topic_id)
        if reps:
            reps = reps[:k]
            emb = embedder.encode(reps, batch_size=64, show_progress_bar=False)
