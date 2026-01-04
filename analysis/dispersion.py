
import numpy as np



def get_centroid_dispersion_per_day(df, embeddings):
    """
    Calculates the dispersion from the centroids in each day.
    """
    daily_dispersion = []
    groups = df.groupby(df["norm_date"])
    for gdate, gdf in groups:
        # Find local positions for embeddings
        local_idx = df.index.get_indexer(gdf.index)        # positions within df_topic    
        emb_day = embeddings[local_idx]  # Get embeddings for this date
        if emb_day.size == 0:
            continue
        centroid_sbert = emb_day.mean(axis=0)  # Compute centroid
        
        daily_dispersion.append(centroid_dispersion(emb_day, centroid_sbert))
    
    return np.array(daily_dispersion)


def centroid_dispersion(emb_day, centroid=None):
    """
    Computes the dispersion as the variance of embeddings around the centroid.
    """
    if centroid is None:
        centroid = emb_day.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    distances = [1 - np.dot(v, centroid) for v in emb_day]
    return np.var(distances)  # or np.mean(distances) for average spread



def within_day_dispersion_variance(emb_day):
    """
    Computes the dispersion within the day of embeddings on a pairwise basis.
    """
    # emb_day: (n_d, d) normalized embeddings
    sims = emb_day @ emb_day.T  # cosine similarity matrix
    upper = sims[np.triu_indices(len(emb_day), k=1)]
    distances = 1 - upper  # cosine distance
    return np.var(distances)  # variance of pairwise distances



import numpy as np
import pandas as pd

def compute_curvature(
    centroids: pd.Series,
    *,
    output_unit: str = "radians",
    use_ema_if_available: bool = True,
    min_norm: float = 1e-9
) -> pd.Series:
    """
    Compute curvature (turning angle) of a centroid trajectory over time.

    Curvature on day t is defined as the angle between the displacement vectors:
        u_{t-1} = c_t - c_{t-1}
        u_{t}   = c_{t+1} - c_{t}
    We then use the cosine rule:
        angle_t = arccos( (u_{t-1} · u_{t}) / (||u_{t-1}|| * ||u_{t}||) )
    The series is aligned to the input index; curvature is only defined
    where both u_{t-1} and u_t exist (i.e., for t in [1 .. T-2]).

    Args:
        centroids: pd.Series whose values are 1D numpy arrays (embedding vectors),
                   ordered by time (e.g., cent_df["centroid_ema"]).
        output_unit: "radians" or "degrees" for the returned curvature values.
        use_ema_if_available: if you pass a DataFrame instead of a Series, and it has
                              "centroid_ema", it will use that column automatically.
        min_norm: numerical floor to avoid division by zero when displacement norms are tiny.

    Returns:
        pd.Series of curvature values aligned to centroids.index:
            - NaN at the first and last positions (cannot form two displacements),
            - NaN where displacement norms are ~0,
            - angle in radians (default) or degrees per `output_unit`.
    """
    # If user passed a DataFrame by accident, pick the right column
    if isinstance(centroids, pd.DataFrame):
        if use_ema_if_available and "centroid_ema" in centroids.columns:
            centroids = centroids["centroid_ema"]
        elif "emb" in centroids.columns:
            centroids = centroids["emb"]
        else:
            raise ValueError("Please provide a Series of centroid vectors or a DataFrame with 'centroid_ema'/'emb'.")

    # Ensure values are numpy arrays
    Cs = [np.asarray(v, dtype=np.float32) if v is not None else None for v in centroids.values]
    idx = centroids.index
    n = len(Cs)

    # Prepare output
    curvature_vals = np.full(n, np.nan, dtype=np.float32)

    # Compute displacements u_{t-1} and u_t, then turning angle at t
    for t in range(1, n - 1):
        c_prev, c_curr, c_next = Cs[t - 1], Cs[t], Cs[t + 1]
        if c_prev is None or c_curr is None or c_next is None:
            curvature_vals[t] = np.nan
            continue

        u_prev = c_curr - c_prev
        u_curr = c_next - c_curr
        n_prev = np.linalg.norm(u_prev)
        n_curr = np.linalg.norm(u_curr)

        # If either displacement is too small, angle is undefined
        if n_prev < min_norm or n_curr < min_norm:
            curvature_vals[t] = np.nan
            continue

        # Cosine of angle; clip for numerical safety
        cos_theta = float(np.dot(u_prev, u_curr) / (n_prev * n_curr))
        cos_theta = max(-1.0, min(1.0, cos_theta))

        angle = np.arccos(cos_theta)  # radians
        curvature_vals[t] = angle

    # Convert unit if requested
    if output_unit.lower() == "degrees":
        curvature_vals = np.degrees(curvature_vals)
    elif output_unit.lower() != "radians":
        raise ValueError("output_unit must be 'radians' or 'degrees'.")

    return pd.Series(curvature_vals, index=idx, name=f"curvature_{output_unit}")
