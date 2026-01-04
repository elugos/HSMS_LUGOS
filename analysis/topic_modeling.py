import hdbscan, umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter
import pandas as pd
import re


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def init_bertopic_model(keywords, seed_multiplier=1.0,
                       umap_params={"n_neighbors": 15, "min_dist": 0.1},
                       hdbscan_params={"min_cluster_size": 30, "min_samples": 10},
                       nr_topics=8,
                       ):
    """
    Fits a BERTopic model to given `texts` and `embeddings` using `keywords` as seed and seed_multiplier.
    """


    # Tune these

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),                 # capture bigrams, e.g.: "mass shooting", "new year"
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # 3 characters or more
        min_df=2,
    )


    # load c-TF-IDF
    ctfidf_model = ClassTfidfTransformer(
        seed_words = keywords,
        seed_multiplier=seed_multiplier,  # Increase to 1.1 or 1.2 if needed
        reduce_frequent_words=True,
    )

    umap_model = umap.UMAP(**umap_params, n_components=10, metric="cosine", random_state=1)
    # hdbscan_model = HDBSCAN(min_cluster_size=25, metric="cosine",
    #                         cluster_selection_epsilon=0.0)
    # WARNING: Use scikit-learn's HDBSCAN will not return `probs` in the right way!

    # Note: we will use Euclidean distance with HDBSCAN because the HDBSCAN package does not implement cosine
    # However, this is fine because the SentenceTransform embeddings are normalized, hence Euclidean is related to Cosine
    hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params, 
                                    metric="euclidean",
                                    cluster_selection_epsilon=0.0,   # Smaller value produces more (and smaller) clusters.
                                    cluster_selection_method="eom",
                                    prediction_data=True)

    topic_model = BERTopic(
                        vectorizer_model=vectorizer, 
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        ctfidf_model=ctfidf_model,
                        calculate_probabilities=True,
                        nr_topics=8,
                        verbose=False)

    return topic_model


def print_topic_coverage_of_seeds(df, topics, topic_model, keywords):
    """
    Prints and plots the topic coverage of seeds.
    """
    seed_set = set(keywords)
    df["topic"] = topics  # ensure topic column present/updated
    topic_ids = sorted([t for t in set(topics) if t != -1])

    topic_seed_stats = []
    for t in topic_ids:
        g = df[df["topic"] == t]

        # Aggregate seed counts across documents
        agg_counter = Counter()
        for d in g["seed_hits_detail"]:
            agg_counter.update(d)

        n_docs = len(g)
        n_docs_with_seed = int((g["seed_hits_count"] > 0).sum())
        prop_docs_with_seed = (n_docs_with_seed / n_docs) if n_docs > 0 else np.nan
        total_seed_hits = sum(agg_counter.values())

        # Top seeds contributing in this topic
        top_seeds = agg_counter.most_common(20)

        topic_seed_stats.append({
            "Topic": t,
            "n_docs_in_topic": n_docs,
            "n_docs_with_seed_hit": n_docs_with_seed,
            "prop_docs_with_seed_hit": prop_docs_with_seed,
            "total_seed_hits": total_seed_hits,
            "top_seeds_in_topic": top_seeds,
        })

    topic_seed_df = pd.DataFrame(topic_seed_stats).sort_values("n_docs_in_topic", ascending=False)
    print(topic_seed_df.head(10))

    def topic_seed_coverage(topic_id, top_n=10):
        # BERTopic returns list of (term, score)
        words_scores = topic_model.get_topic(topic_id)
        topic_words = [w for (w, _) in (words_scores or [])][:top_n]
        topic_words_norm = [re.sub(r'[^a-z\s]', '', w.lower()).strip() for w in topic_words]
        coverage = sum(1 for w in topic_words_norm if w in seed_set) / max(1, len(topic_words_norm))
        return topic_words, coverage

    coverage_rows = []
    for t in topic_ids:
        tw, cov = topic_seed_coverage(t, top_n=15)
        coverage_rows.append({"Topic": t, "top_words": tw, "seed_coverage": cov})

    topic_seed_coverage_df = pd.DataFrame(coverage_rows).sort_values("seed_coverage", ascending=False)
    print(topic_seed_coverage_df.head(10))


    plt.figure(figsize=(8,5))
    sns.barplot(
        data=topic_seed_df.head(15),
        x="Topic", y="prop_docs_with_seed_hit",
        order=topic_seed_df.head(15)["Topic"]
    )
    plt.title("Proportion of docs with ≥1 seed hit per topic")
    plt.ylabel("Proportion")
    plt.xlabel("Topic")
    plt.tight_layout()
    plt.show()

    # Example: show top 5 docs (by seed hit count) in largest topic
    largest_topic_id = int(topic_seed_df.iloc[0]["Topic"])
    inspect = df[df["topic"] == largest_topic_id].copy()
    inspect = inspect.sort_values("seed_hits_count", ascending=False).head(5)
    print(inspect[["date", "title", "seed_hits_count", "seed_hits_unique", "seed_hits_detail"]])


def reassign_outliers(df, probs, reassign_prob=0.25):
    reassign_prob = 0.25

    # Reassign based on HDBSCAN probability scores
    top_prob = np.max(probs, axis=1)
    top_topic = np.argmax(probs, axis=1)

    reassign_mask = (df["topic"] == -1)
    reassign_candidates = reassign_mask.sum()
    print(f"Outliers before reassign: {reassign_candidates}")
    df.loc[reassign_mask & (top_prob > reassign_prob), "topic"] = top_topic[reassign_mask & (top_prob > reassign_prob)]

    print(f"Outliers after reassign: {(df['topic']==-1).sum()}")




from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np

def knn_majority_flip(
    embeddings: np.ndarray,
    topic_mask: np.ndarray,
    k: int = 10,
    vote_ratio: float = 0.6,
    similarity_threshold: Optional[float] = None,
    normalize: bool = True,
    neighbor_pool: Literal["all", "topic_only"] = "all",
    exclude_mask: Optional[np.ndarray] = None,
    require_min_topic_neighbors: int = 3,
    cap_additions: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    For each non-topic document, compute its k nearest neighbors and flip to topic
    if the majority of neighbors are topic documents.

    Args:
        embeddings: (n_docs, dim) SBERT embeddings. If not unit-normalized, set `normalize=True`.
        topic_mask: (n_docs,) boolean array; True for docs currently marked as topic.
        k: number of neighbors to query per document.
        vote_ratio: minimum proportion of neighbors that must be topic to flip (e.g., 0.6).
        similarity_threshold: optional cosine similarity cutoff; neighbors below are ignored
                              for voting (and excluded from counts). Typical range: 0.30–0.45.
        normalize: L2-normalize embeddings before similarity computations.
        neighbor_pool: 
            "all"        -> neighbors drawn from the whole corpus (default).
            "topic_only" -> neighbors drawn only from current topic docs (stronger vote).
        exclude_mask: optional (n_docs,) boolean array marking docs that must NEVER be flipped.
        require_min_topic_neighbors: minimum absolute count of topic neighbors required to flip,
                                     after applying `similarity_threshold`.
        cap_additions: optional cap on how many non-topic docs can be flipped (highest-confidence first).

    Returns:
        augmented_mask: (n_docs,) boolean mask with flips applied.
        details: dict keyed by doc id (only non-topic docs processed), each containing:
                 {
                   'neighbor_ids': np.ndarray,
                   'neighbor_sims': np.ndarray,
                   'topic_neighbor_count': int,
                   'topic_neighbor_ratio': float,
                   'flipped': bool
                 }
    """
    n_docs = int(embeddings.shape[0])
    topic_mask = np.asarray(topic_mask, dtype=bool)
    if exclude_mask is None:
        exclude_mask = np.zeros(n_docs, dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)

    # --- Normalize embeddings to unit length for cosine via dot product ---
    E = np.asarray(embeddings, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.clip(norms, 1e-12, None)

    # --- Neighbor search set ---
    if neighbor_pool == "topic_only":
        pool_mask = topic_mask & (~exclude_mask)
    else:  # "all"
        pool_mask = (~exclude_mask)  # allow both topic & non-topic in the pool

    pool_ids = np.where(pool_mask)[0]
    if pool_ids.size == 0:
        # Nothing to vote with; return unchanged
        return topic_mask.copy(), {}

    # Prepare outputs
    augmented_mask = topic_mask.copy()
    details: Dict[int, Dict[str, Any]] = {}

    # Build fast neighbor index (scikit-learn if available), else NumPy fallback
    use_sklearn = True
    try:
        from sklearn.neighbors import NearestNeighbors
        use_sklearn = True
    except Exception:
        use_sklearn = False

    # Precompute for sklearn path
    if use_sklearn:
        # Cosine distance = 1 - cosine similarity; we trained on normalized vectors
        nn = NearestNeighbors(n_neighbors=min(k + 1, pool_ids.size), metric="cosine", algorithm="auto")
        nn.fit(E[pool_ids])

    # Candidate non-topic docs
    non_topic_ids = np.where((~topic_mask) & (~exclude_mask))[0]
    if non_topic_ids.size == 0:
        return augmented_mask, {}

    # Collect flips with confidence to enforce cap_additions later
    flip_candidates = []

    for doc_id in non_topic_ids:
        # Skip if doc is not eligible
        if exclude_mask[doc_id]:
            continue

        # --- Query neighbors ---
        if use_sklearn:
            # Query against the pool index
            dists, idxs_rel = nn.kneighbors(E[doc_id][None, :], n_neighbors=min(k + 1, pool_ids.size), return_distance=True)
            dists = dists[0]  # (m,)
            idxs_rel = idxs_rel[0]  # (m,)
            neighbor_ids = pool_ids[idxs_rel]  # absolute ids in corpus

            # Drop self if present
            self_pos = np.where(neighbor_ids == doc_id)[0]
            if self_pos.size > 0:
                neighbor_ids = np.delete(neighbor_ids, self_pos[0])
                dists = np.delete(dists, self_pos[0])

            # Keep top-k after removing self
            neighbor_ids = neighbor_ids[:k]
            dists = dists[:k]
            # Convert to cosine similarity
            sims = (1.0 - dists).astype(np.float32)
        else:
            # NumPy fallback: dense cosine via dot product
            sims_full = E @ E[doc_id]  # (n_docs,)
            sims = sims_full[pool_ids]
            neighbor_order = np.argsort(sims)[::-1]  # descending sims
            neighbor_ids = pool_ids[neighbor_order]

            # Drop self if present
            if neighbor_ids.size > 0 and neighbor_ids[0] == doc_id:
                neighbor_ids = neighbor_ids[1:]
                sims = sims[neighbor_order][1:]
            else:
                sims = sims[neighbor_order]

            # Keep top-k
            neighbor_ids = neighbor_ids[:k]
            sims = sims[:k]

        if neighbor_ids.size == 0:
            # No neighbors (or only self), no flip
            details[doc_id] = {
                "neighbor_ids": np.array([], dtype=int),
                "neighbor_sims": np.array([], dtype=np.float32),
                "topic_neighbor_count": 0,
                "topic_neighbor_ratio": 0.0,
                "flipped": False,
            }
            continue

        # --- Apply similarity threshold (if provided) ---
        if similarity_threshold is not None:
            keep = sims >= float(similarity_threshold)
            neighbor_ids = neighbor_ids[keep]
            sims = sims[keep]

        # --- Count topic neighbors ---
        topic_neighbors = topic_mask[neighbor_ids]
        topic_count = int(np.count_nonzero(topic_neighbors))
        denom = max(1, neighbor_ids.size)  # avoid divide-by-zero after thresholding
        ratio = topic_count / denom

        # --- Decide flip ---
        flipped = (
            (ratio >= vote_ratio) and
            (topic_count >= require_min_topic_neighbors)
        )

        details[doc_id] = {
            "neighbor_ids": neighbor_ids,
            "neighbor_sims": sims,
            "topic_neighbor_count": topic_count,
            "topic_neighbor_ratio": ratio,
            "flipped": bool(flipped),
        }

        if flipped:
            # Store candidate with a confidence proxy -> ratio * mean(similarity)
            conf = float(ratio) * (float(np.mean(sims)) if sims.size > 0 else 0.0)
            flip_candidates.append((doc_id, conf))

    # --- Optionally cap the number of additions ---
    if cap_additions is not None and cap_additions >= 0:
        flip_candidates.sort(key=lambda x: x[1], reverse=True)  # by confidence
        for i, (doc_id, _) in enumerate(flip_candidates):
            if i < cap_additions:
                augmented_mask[doc_id] = True
            else:
                details[doc_id]["flipped"] = False  # mark as not flipped due to cap
    else:
        for doc_id, _ in flip_candidates:
            augmented_mask[doc_id] = True

    return augmented_mask, details
