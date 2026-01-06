
import numpy as np
import pandas as pd

def preprocess_signal(s: pd.Series, ema_alpha=0.2, zscore=False):
    """EMA-smooth + (optional) z-score standardization; returns np.ndarray."""
    x = s.to_numpy(dtype=float)
    # Fill small gaps
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    # EMA smoothing
    ema = []
    prev = None
    for v in x:
        prev = v if prev is None else (ema_alpha * v + (1 - ema_alpha) * prev)
        ema.append(prev)
    x_sm = np.array(ema, dtype=float)
    # Robust z-score
    if zscore:
        mu = np.nanmedian(x_sm)
        mad = np.nanmedian(np.abs(x_sm - mu)) + 1e-12
        x_sm = (x_sm - mu) / (1.4826 * mad)  # robust z
    return x_sm



def get_zscores(x, window=3):
    """
    Return the z-scores for given series.
    """
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=1).mean()
    sd = s.rolling(window, min_periods=1).std().replace(0, np.nan)
    zscore = ((s-mu)/sd)
    return zscore


def cp_candidates_zscore(x, window=3, z="auto", min_gap=2, z_perc=95):
    """
    Flag points whose z-score deviates from rolling mean by >= z.
    z_perc is used for z="auto".
    Returns sorted indices of candidate change points.
    """
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=1).mean()
    sd = s.rolling(window, min_periods=1).std().fillna(0.0)
    zscore = (s - mu) / (sd + 1e-6)

    if z == "auto":
        z = np.percentile(zscore, z_perc)  # Find z that is in the n-th percentile
    idx = np.where(np.abs(zscore) >= z)[0]
    # enforce minimum gap between CPs
    cp = []
    last = -10**9
    for i in idx:
        if i - last >= min_gap:
            cp.append(i); last = i
    return cp



def binary_segmentation_mean(x, min_seg_len=3, penalty=1.0):
    """
    Piecewise-constant mean with L2 loss. penalty controls CP count (higher = fewer CPs).
    Returns sorted list of change point indices (segment end indices).
    """
    n = len(x)
    cps = []
    def cost(a, b):
        seg = x[a:b]
        mu = np.mean(seg)
        return np.sum((seg - mu)**2)

    def recurse(a, b):
        if b - a < 2 * min_seg_len:
            return
        base = cost(a, b)
        best_gain, best_t = 0.0, None
        for t in range(a + min_seg_len, b - min_seg_len + 1):
            left = cost(a, t); right = cost(t, b)
            gain = base - (left + right) - penalty
            if gain > best_gain:
                best_gain, best_t = gain, t
        if best_t is not None:
            cps.append(best_t)
            recurse(a, best_t)
            recurse(best_t, b)

    recurse(0, n)
    cps.sort()
    return cps



def cusum_cp(x, k=0.5, h=3.0, min_gap=3):
    """
    Basic CUSUM on standardized series x; k is reference drift, h threshold.
    Returns CP indices.
    """
    s_pos = 0.0; s_neg = 0.0
    cp = []; last = -10**9
    for i, xi in enumerate(x):
        s_pos = max(0.0, s_pos + xi - k)
        s_neg = min(0.0, s_neg + xi + k)
        if s_pos > h or s_neg < -h:
            if i - last >= min_gap:
                cp.append(i); last = i
            s_pos = 0.0; s_neg = 0.0
    return cp



import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

def percentile_bursts(x: pd.Series, q=95, window=3, min_gap=2,
                      k=1.0,
                      location="median"):
    """
    Compute the percentile bursts.
    q is the threshold percentile, window is the rolling window for median calculation,
    k is the MAD adjustment factor.
    """
    # EMA smooth to reduce day-to-day noise; no z-score, just quantiles
    # ema = []
    # # prev = None
    # # for v in x.fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float):
    # #     prev = v if prev is None else 0.25*v + 0.75*prev
    # #     ema.append(prev)
    # xs = np.array(ema, dtype=float)

    # rolling median & MAD (mean absolute deviation) (robust baseline)
    s = pd.Series(x)

    if location == "median":
        # Rolling median
        loc = s.rolling(window, min_periods=1).median()

        # # Manual implementation of MAD (median)
        abs_dev = np.abs(s-loc)
        mad = abs_dev.rolling(window, min_periods=1).median()
        scale = pd.Series(mad, index=s.index).clip(lower=1e-6) * 1.4826
    elif location == "mean":
        # Rolling mean
        loc = s.rolling(window, min_periods=1).mean()
        abs_dev = np.abs(s-loc)
        mae = abs_dev.rolling(window, min_periods=1).mean()
        scale = pd.Series(mae, index=s.index).clip(lower=1e-6)
    elif location == "mad":  # Median Absolute Deviation
        # Rolling MAD
        loc = s.rolling(window, min_periods=1).median()
        mad = np.abs(s-loc).median()
        scale = k*mad
        

    # Using scipy for MAD
    # mad = median_abs_deviation(s, scale=1.4826)

    # Response scores
    rscore = np.abs((s - loc)) / scale  # robust standardized residuals (robust statistics)
    thr = np.nanpercentile(rscore, q)
    idx = np.where(rscore >= thr)[0]

    # min-gap to avoid clustered duplicates
    cps, last = [], -10**9
    for i in idx:
        if i - last >= min_gap:
            cps.append(int(i)); last = i
    return cps, rscore, loc, thr


def detect_cp_abs_dev(x, mad_factor=3, high_percentile=95, location="mean"):
    x = np.array(x)

    if location == "mean":
        mean_val = np.mean(x)
        loc = np.mean(np.abs(x - mean_val))  # Mean Absolute Deviation
    elif location == "median":
        median_val = np.median(x)
        loc = np.mean(np.abs(x - median_val))
    
    # Compute threshold using percentile + MAD_mean
    p_high = np.percentile(x, high_percentile)
    threshold = p_high + mad_factor * loc
    
    # Identify bursts
    burst_indices = np.where(x > threshold)[0]
    
    # Group consecutive indices into segments
    cps = []
    if len(burst_indices) > 0:
        start = burst_indices[0]
        for i in range(1, len(burst_indices)):
            if burst_indices[i] != burst_indices[i-1] + 1:
                cps.append((start, burst_indices[i-1]))
                start = burst_indices[i]
        cps.append((start, burst_indices[-1]))
    
    return cps, threshold

