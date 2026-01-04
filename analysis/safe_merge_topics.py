### MERGE USING SEED HITS


import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from sentence_transformers import util

# -----------------------------
# 1) Build topic-level seed vectors with IDF weighting
# -----------------------------
def compute_topic_seed_vectors(
    df: pd.DataFrame,
    seed_words: List[str],
    topic_col: str = "topic"
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], np.ndarray]:
    """
    Returns:
      topic_vecs: dict {topic_id: (len(seed_words),) np.ndarray} weighted by topic-IDF
      topic_sizes: dict {topic_id: n_docs}
      idf: (len(seed_words),) np.ndarray of IDF weights across topics
    """
    # Map seed index
    seed_index = {w: i for i, w in enumerate(seed_words)}
    # Collect topics (exclude -1 unless you want to merge outliers too)
    topic_ids = sorted(t for t in set(df[topic_col]) if t != -1)

    # Per-topic raw seed counts
    raw_counts = {t: np.zeros(len(seed_words), dtype=np.float32) for t in topic_ids}
    topic_sizes = {t: int((df[topic_col] == t).sum()) for t in topic_ids}

    # Count occurrences per topic
    for t in topic_ids:
        g = df[df[topic_col] == t]["seed_hits_detail"]
        # g consists of dicts {seed: freq} per doc
        agg = Counter()
        for d in g:
            if isinstance(d, dict) and d:
                agg.update(d)
        vec = raw_counts[t]
        for w, c in agg.items():
            if w in seed_index:
                vec[seed_index[w]] += float(c)

    # Topic presence per seed (for IDF): in how many topics does each seed appear?
    presence = np.zeros(len(seed_words), dtype=np.int32)
    for j in range(len(seed_words)):
        for t in topic_ids:
            if raw_counts[t][j] > 0:
                presence[j] += 1

    # Smooth IDF (topics-based)
    # idf_j = log( (T + 1) / (1 + presence_j) ) + 1
    T = max(1, len(topic_ids))
    idf = np.log((T + 1) / (1 + presence)) + 1.0

    # Apply IDF and L2-normalize
    topic_vecs = {}
    for t in topic_ids:
        v = raw_counts[t] * idf
        # Optional: normalize by topic size to get per-doc influence
        if topic_sizes[t] > 0:
            v = v / topic_sizes[t]
        # L2 normalize to compare with cosine
        n = np.linalg.norm(v)
        topic_vecs[t] = v if n == 0 else (v / n)

    return topic_vecs, topic_sizes, idf


# -----------------------------
# 2) (Optional) representative embedding similarity gate
# -----------------------------
def topic_rep_embedding(topic_model, embedder, topic_id, k=10):
    rep_docs = topic_model.get_representative_docs(topic_id)[:k]
    if len(rep_docs) == 0:
        return None
    emb = embedder.encode(rep_docs, batch_size=64, show_progress_bar=False)
    v = emb.mean(axis=0)
    # Normalize
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def build_embed_index(topic_model, embedder, topic_ids: List[int]) -> Dict[int, np.ndarray]:
    idx = {}
    for t in topic_ids:
        v = topic_rep_embedding(topic_model, embedder, t)
        if v is not None:
            idx[t] = v
    return idx


# -----------------------------
# 3) Build candidate merges by seed similarity (and optional embedding gate)
# -----------------------------
def build_seed_merge_pairs(
    topic_vecs: Dict[int, np.ndarray],
    topic_sizes: Dict[int, int],
    seed_sim_threshold: float = 0.60,   # cosine in seed space
    min_shared_seeds: int = 2,          # require at least 2 non-zero seed dimensions in each topic
    # Embedding gate
    embed_index: Dict[int, np.ndarray] = None,
    require_embedding_sim: bool = True,
    embed_sim_threshold: float = 0.65,
    # Size constraints
    max_merge_ratio: float = 3.0,       # do not merge if larger/smaller ratio too extreme
) -> List[Tuple[int, int]]:
    """
    Returns pairs (ti, tj) to merge. We do not apply connected-component logic here yet;
    we just propose edges that pass criteria.
    """
    tids = sorted(topic_vecs.keys())
    pairs = []

    # Precompute non-zero seed counts per topic
    nnz = {t: int(np.count_nonzero(topic_vecs[t])) for t in tids}

    for i, ti in enumerate(tids):
        for tj in tids[i+1:]:
            # Basic nnz constraint
            if nnz[ti] < min_shared_seeds or nnz[tj] < min_shared_seeds:
                continue

            # Seed cosine similarity
            s = float(np.dot(topic_vecs[ti], topic_vecs[tj]))
            if s < seed_sim_threshold:
                continue

            # Size ratio constraint
            a, b = topic_sizes[ti], topic_sizes[tj]
            if a == 0 or b == 0:
                continue
            r = max(a, b) / max(1, min(a, b))
            if r > max_merge_ratio:
                # Skip extremely imbalanced merges unless both are small (you can relax this if needed)
                continue

            # Optional embedding gate
            if require_embedding_sim and embed_index is not None and (ti in embed_index) and (tj in embed_index):
                e = util.cos_sim(embed_index[ti], embed_index[tj]).item()
                if e < embed_sim_threshold:
                    continue

            pairs.append((ti, tj))

    return pairs


# -----------------------------
# 4) Merge by connected components (merge all topics in each component into the largest)
# -----------------------------
def connected_components_from_pairs(pairs: List[Tuple[int, int]]) -> List[List[int]]:
    # Build adjacency
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    # Gather nodes
    nodes = set()
    for a, b in pairs:
        nodes.add(a); nodes.add(b)
    # DFS
    comps = []
    visited = set()
    for s in nodes:
        if s in visited:
            continue
        stack = [s]
        comp = []
        visited.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def merge_topics_using_seed_hits(
    df: pd.DataFrame,
    seed_words: List[str],
    topic_model,
    texts: List[str],
    embeddings: np.ndarray,
    embedder=None,
    rounds: int = 2,
    seed_sim_threshold: float = 0.60,
    embed_sim_threshold: float = 0.75,
    min_shared_seeds: int = 3,
    max_merge_ratio: float = 3.0,
    require_embedding_sim: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Runs up to `rounds` of seed-based merging. After each round, transforms to update assignments.
    Returns updated topic_info DataFrame.
    """
    for r in range(rounds):
        # Recompute vectors on current assignments
        topic_vecs, topic_sizes, idf = compute_topic_seed_vectors(df, seed_words, topic_col="topic")
        topic_ids = sorted(topic_vecs.keys())

        # Optional: embedding gate
        embed_index = None
        if require_embedding_sim and (embedder is not None) and len(topic_ids) > 0:
            embed_index = build_embed_index(topic_model, embedder, topic_ids)

        # Build candidate edges
        edges = build_seed_merge_pairs(
            topic_vecs=topic_vecs,
            topic_sizes=topic_sizes,
            seed_sim_threshold=seed_sim_threshold,
            min_shared_seeds=min_shared_seeds,
            embed_index=embed_index,
            require_embedding_sim=require_embedding_sim,
            embed_sim_threshold=embed_sim_threshold,
            max_merge_ratio=max_merge_ratio,
        )

        if verbose:
            print(f"[Round {r+1}] Candidate edges (seed-based): {len(edges)}")

        if not edges:
            break

        # Connected components over edges
        comps = connected_components_from_pairs(edges)
        if verbose:
            print(f"[Round {r+1}] Components to merge: {len(comps)}")

        # For each component, merge all topics into the largest one
        pairs_to_merge = []
        for comp in comps:
            if len(comp) < 2:
                continue
            # Representative = largest by n_docs
            rep = max(comp, key=lambda t: topic_sizes.get(t, 0))
            for t in comp:
                if t != rep:
                    pairs_to_merge.append((rep, t))

        if verbose:
            print(f"[Round {r+1}] Calling BERTopic.merge_topics with {len(pairs_to_merge)} pairs...")
            print(pairs_to_merge[:10])

        if len(pairs_to_merge) == 0:
            break

        # Merge in BERTopic
        topic_model.merge_topics(texts, topics_to_merge=pairs_to_merge)

        # Update assignments after merge
        new_topics, new_probs = topic_model.transform(texts, embeddings)
        df["topic"] = new_topics

    # Return final topic info
    topic_info = topic_model.get_topic_info()
    if verbose:
        print(topic_info.head(15))
    return topic_info
