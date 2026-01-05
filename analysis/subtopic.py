
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

def _ensure_unit(E: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(norms, 1e-12, None)

def subtopic_coherence_embedding(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    day_col: str = "norm_date",
    topic_col: str = "topic",
    exclude_topic_id: int = -1,
    normalize: bool = True,
    compute_silhouette: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per (day, subtopic): compute mean cosine similarity to centroid and
    variance of cosine distances; plus day-level aggregates & silhouette score.

    Returns:
      per_topic_day: DataFrame with columns
        [day, topic_id, n_docs, mean_sim_to_centroid, var_dist_to_centroid]
      per_day_agg: DataFrame with columns
        [day, n_docs_day, n_topics_active, weighted_mean_sim, weighted_mean_var,
         silhouette (optional)]
    """
    if normalize:
        E = _ensure_unit(np.asarray(embeddings, dtype=np.float32))
    else:
        E = np.asarray(embeddings, dtype=np.float32)

    # Attach row positions to df for alignment
    df = df.copy()
    df["row_id"] = np.arange(len(df))

    per_topic_day_rows = []
    per_day_rows = []

    # Iterate per day
    for day, g in df.groupby(day_col):
        # Filter out outliers
        g = g[g[topic_col] != exclude_topic_id]
        idx_day = g["row_id"].to_numpy()
        # if idx_day.size == 0:
        #     per_day_rows.append({
        #         "day": day, "n_docs_day": 0, "n_topics_active": 0,
        #         "weighted_mean_sim": np.nan, "weighted_mean_var": np.nan,
        #         "silhouette": np.nan if compute_silhouette else None
        #     })
        #     continue

        E_day = E[idx_day]
        # Per-subtopic coherence
        stats = []
        for topic_id, gt in g.groupby(topic_col):
            idx_t = gt["row_id"].to_numpy()
            E_t = E[idx_t]
            n_t = E_t.shape[0]
            if n_t == 0:
                continue
            # Subtopic centroid (mean)
            c_t = E_t.mean(axis=0)
            c_t /= np.linalg.norm(c_t) + 1e-12
            sims = E_t @ c_t  # cosine similarity to centroid
            dists = 1.0 - sims
            mean_sim = float(np.mean(sims))
            var_dist = float(np.var(dists))
            stats.append((topic_id, n_t, mean_sim, var_dist))

            per_topic_day_rows.append({
                "day": day,
                "topic": topic_id,
                "n_docs": n_t,
                "mean_sim_to_centroid": mean_sim,
                "var_dist_to_centroid": var_dist
            })

        if not stats:
            per_day_rows.append({
                "day": day, "n_docs_day": int(len(idx_day)), "n_topics_active": 0,
                "weighted_mean_sim": np.nan, "weighted_mean_var": np.nan,
                "silhouette": np.nan if compute_silhouette else None
            })
            continue

        # Day-level weighted aggregates across subtopics
        n_docs_t = np.array([s[1] for s in stats], dtype=float)
        mean_sim_t = np.array([s[2] for s in stats], dtype=float)
        var_dist_t = np.array([s[3] for s in stats], dtype=float)

        w_mean_sim = float(np.average(mean_sim_t, weights=n_docs_t))
        w_mean_var = float(np.average(var_dist_t, weights=n_docs_t))
        n_topics_active = int(len(stats))
        n_docs_day = int(len(idx_day))

        # Optional silhouette score across all subtopics present that day
        if compute_silhouette and n_topics_active >= 2 and n_docs_day >= 5:
            try:
                from sklearn.metrics import silhouette_score
                labels_day = g[topic_col].to_numpy()
                sil = float(silhouette_score(E_day, labels_day, metric="cosine"))
            except Exception:
                sil = np.nan
        else:
            sil = np.nan

        per_day_rows.append({
            "day": day,
            "n_docs_day": n_docs_day,
            "n_topics_active": n_topics_active,
            "weighted_mean_sim": w_mean_sim,
            "weighted_mean_var": w_mean_var,
            "silhouette": sil
        })

    per_topic_day = pd.DataFrame(per_topic_day_rows).sort_values(["day","topic"]).reset_index(drop=True)
    per_day_agg = pd.DataFrame(per_day_rows).sort_values("day").reset_index(drop=True)
    return per_topic_day, per_day_agg



import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    denom = len(sa | sb)
    return 0.0 if denom == 0 else len(sa & sb) / denom

def day_lexical_coherence(
    df: pd.DataFrame,
    topic_model,
    day_col: str = "norm_date",
    topic_col: str = "topic",
    exclude_topic_id: int = -1,
    top_n_terms: int = 10
) -> pd.DataFrame:
    """
    For each day, compute mean pairwise Jaccard similarity of top terms
    across subtopics active that day (from BERTopic's c-TF-IDF).
    """
    rows = []
    for day, g in df.groupby(day_col):
        topics_active = sorted(t for t in set(g[topic_col]) if t != exclude_topic_id)
        if len(topics_active) < 2:
            rows.append({"day": day, "lexical_jaccard_mean": np.nan, "n_topics_active": len(topics_active)})
            continue

        termsets: Dict[int, List[str]] = {}
        for t in topics_active:
            tw = topic_model.get_topic(t) or []  # list of (term, score)
            termsets[t] = [w for (w, _) in tw[:top_n_terms]]

        # Pairwise Jaccard among subtopics
        sims = []
        for i, ti in enumerate(topics_active):
            for tj in topics_active[i+1:]:
                sims.append(jaccard_similarity(termsets[ti], termsets[tj]))

        rows.append({
            "day": day,
            "lexical_jaccard_mean": float(np.mean(sims)) if sims else np.nan,
            "n_topics_active": len(topics_active)
        })

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
