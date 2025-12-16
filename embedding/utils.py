import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cameo_codes():
    cameo_codes = {
    "MAKE PUBLIC STATEMENT": "1",
    "APPEAL": "2",
    "EXPRESS INTENT TO COOPERATE": "3",
    "CONSULT": "4",
    "ENGAGE IN DIPLOMATIC COOPERATION": "5",
    "ENGAGE IN MATERIAL COOPERATION": "6",
    "PROVIDE AID": "7",
    "YIELD": "8",
    "INVESTIGATE": "9",
    "DEMAND": "10",
    "DISAPPROVE": "11",
    "REJECT": "12",
    "THREATEN": "13",
    "PROTEST": "14",
    "EXHIBIT FORCE POSTURE": "15",
    "COERCE": "16",
    "ASSAULT": "17",
    "FIGHT": "19",
    "USE UNCONVENTIONAL MASS VIOLENCE": "20",
    }
    return cameo_codes



def vectorize(series):
    """
    Takes a pandas Series of embedding_json strings
    and returns a NumPy array of shape (N, 384).
    """
    # Ensure string
    s = series.astype(str).str.strip()

    # Remove brackets, split into float lists
    vecs = (
        s.str[1:-1]                   # strip '[' and ']'
         .str.split(',')              # split by comma
         .apply(lambda x: np.array(x, dtype=np.float32))
         .tolist()
    )

    # Stack into (n_samples, embed_dim)
    return np.vstack(vecs)



def avg_pairwise_distance_by_label(x1, x2, labels, metric="cosine"):
    """
    Works with BOTH numpy arrays and torch tensors.
    Computes avg distance for same-label and opposite-label pairs.
    """

    # --- Convert everything to torch tensors if needed ---
    if isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1)
    if isinstance(x2, np.ndarray):
        x2 = torch.from_numpy(x2)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    x1 = x1.float()
    x2 = x2.float()
    labels = labels.int()

    # --- Distance computation ---
    if metric == "cosine":
        x1n = F.normalize(x1, p=2, dim=1)
        x2n = F.normalize(x2, p=2, dim=1)
        distances = 1 - torch.sum(x1n * x2n, dim=1)

    elif metric == "euclidean":
        distances = torch.norm(x1 - x2, dim=1)

    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    distances = distances.cpu().numpy()
    labels = labels.cpu().numpy()

    same_mask = labels == 1
    opp_mask  = labels == 0

    same_dist = distances[same_mask]
    opp_dist = distances[opp_mask]

    avg_same = distances[same_mask].mean() if same_mask.any() else np.nan
    avg_opp  = distances[opp_mask].mean() if opp_mask.any() else np.nan

    return same_dist, opp_dist,{
        "avg_same": float(avg_same),
        "avg_opposite": float(avg_opp),
        "n_same": int(same_mask.sum()),
        "n_opposite": int(opp_mask.sum())
    }