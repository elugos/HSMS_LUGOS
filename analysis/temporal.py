
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def compute_mad_thresholds(
    s: pd.Series,
    window: str = "30D",
    min_periods: int = 7,
    k: float = 5.0
) -> pd.Series:
    """
    Compute rolling MAD-based threshold:
        threshold(t) = rolling_median(t) + k * rolling_MAD(t)
    where rolling_MAD(t) = median(|x - rolling_median|) within the window.

    Parameters
    ----------
    s : pd.Series
        Time series indexed by DatetimeIndex.
    window : str
        Time offset for rolling window (e.g., '30D', '8W').
    min_periods : int
        Minimum points required in a window to compute metrics.
    k : float
        Multiplicative factor for MAD threshold.

    Returns
    -------
    pd.Series
        Threshold series aligned to s.index.
    """
    # Rolling median
    rm = s.rolling(window=window, min_periods=min_periods).median()

    # Rolling MAD (median absolute deviation around rolling median)
    abs_dev = (s - rm).abs()
    rmad = abs_dev.rolling(window=window, min_periods=min_periods).median()

    threshold = rm + k * rmad
    return threshold


def detect_events_mad(
    s: pd.Series,
    window: str = "30D",
    min_periods: int = 7,
    k: float = 5.0,
    min_points: int = 1,
    consolidate_gap: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect events using rolling MAD thresholds. An event is a contiguous region
    where s(t) > threshold(t). Onset is the first point of the region; end is the last.

    Parameters
    ----------
    s : pd.Series
        Time series with DatetimeIndex.
    window : str
        Rolling window for baseline median/MAD.
    min_periods : int
        Minimum points to compute rolling statistics.
    k : float
        MAD multiplier (typical values 3–7).
    min_points : int
        Minimum length of an above-threshold run to be considered an event.
    consolidate_gap : Optional[str]
        If provided (e.g., '2D'), merge two adjacent events if the gap between them
        is less than this duration.

    Returns
    -------
    pd.DataFrame
        Event table with columns:
        ['event_id', 'onset_dt', 'end_dt', 'peak_dt', 'peak_value', 'length']
    """
    thr = compute_mad_thresholds(s, window=window, min_periods=min_periods, k=k)
    above = s > thr

    # Label contiguous runs of above-threshold points
    change = above.ne(above.shift(1)).cumsum()
    runs = (
        pd.DataFrame({
            "group_id": change,
            "dt": s.index,
            "above": above.values,
            "value": s.values
        })
        .groupby(["group_id", "above"])
    )

    event_rows = []
    event_id = 0

    for (gid, is_above), g in runs:
        if not is_above:
            continue
        if len(g) < min_points:
            continue

        # Onset and end
        onset_dt = g["dt"].iloc[0]
        end_dt = g["dt"].iloc[-1]
        # Peak within the event window
        idxmax = g["value"].idxmax()
        # g["dt"] is not the index, so we need to find peak using the original series
        window_mask = (s.index >= onset_dt) & (s.index <= end_dt)
        segment = s[window_mask]
        peak_dt = segment.idxmax()
        peak_value = segment.loc[peak_dt]

        event_rows.append({
            "event_id": event_id,
            "onset_dt": onset_dt,
            "end_dt": end_dt,
            "peak_dt": peak_dt,
            "peak_value": peak_value,
            "length": (end_dt - onset_dt)
        })
        event_id += 1

    events_df = pd.DataFrame(event_rows)

    # Optional consolidation: merge events separated by short gaps
    if consolidate_gap and not events_df.empty:
        consolidated = []
        cur = events_df.iloc[0].to_dict()
        for i in range(1, len(events_df)):
            nxt = events_df.iloc[i].to_dict()
            gap = nxt["onset_dt"] - cur["end_dt"]
            if gap < pd.Timedelta(consolidate_gap):
                # Merge: extend end_dt/peak if needed
                # Keep the earlier onset_dt
                cur["end_dt"] = max(cur["end_dt"], nxt["end_dt"])
                # Update peak
                cur_peak_val = s[(s.index >= cur["onset_dt"]) & (s.index <= cur["end_dt"])].max()
                cur_peak_dt = s[(s.index >= cur["onset_dt"]) & (s.index <= cur["end_dt"])].idxmax()
                cur["peak_dt"] = cur_peak_dt
                cur["peak_value"] = cur_peak_val
                cur["length"] = cur["end_dt"] - cur["onset_dt"]
            else:
                consolidated.append(cur)
                cur = nxt
        consolidated.append(cur)
        events_df = pd.DataFrame(consolidated)
        # Reassign event_id sequentially
        events_df["event_id"] = range(len(events_df))

    return events_df[["event_id", "onset_dt", "end_dt", "peak_dt", "peak_value", "length"]]


def compute_event_baseline(
    s: pd.Series,
    onset_dt: pd.Timestamp,
    baseline_window: str = "30D",
    min_periods: int = 7
) -> float:
    """
    Compute a robust baseline right before the event onset using rolling median.
    Falls back to rolling median at onset if not enough pre-event data.

    Parameters
    ----------
    s : pd.Series
        Time series with DatetimeIndex.
    onset_dt : pd.Timestamp
        Event onset datetime.
    baseline_window : str
        Window to compute baseline median (e.g., '30D').
    min_periods : int
        Minimum points required.

    Returns
    -------
    float
        Baseline value.
    """
    # Use data strictly before onset
    pre_mask = s.index < onset_dt
    pre_series = s[pre_mask]
    if not pre_series.empty:
        baseline = pre_series.rolling(window=baseline_window, min_periods=min_periods).median().iloc[-1]
        if pd.isna(baseline):
            # fallback to rolling median including onset
            baseline = s.rolling(window=baseline_window, min_periods=min_periods).median().loc[onset_dt]
    else:
        baseline = s.rolling(window=baseline_window, min_periods=min_periods).median().loc[onset_dt]
    return float(baseline)


def compute_half_life_threshold(
    s: pd.Series,
    peak_dt: pd.Timestamp,
    baseline: float,
    amplitude: float
) -> Dict[str, Optional[pd.Timestamp]]:
    """
    Threshold-based half-life: find the first time after peak where
    s(t) <= baseline + amplitude/2. Returns half-life time and datetime.

    Parameters
    ----------
    s : pd.Series
        Time series with DatetimeIndex.
    peak_dt : pd.Timestamp
        Datetime of the peak.
    baseline : float
        Baseline value.
    amplitude : float
        Peak - baseline.

    Returns
    -------
    dict
        {'half_life_dt': Timestamp or None, 'half_life_days': float or None}
    """
    target = baseline + amplitude / 2.0
    tail = s[s.index > peak_dt]
    cross_mask = tail <= target
    if cross_mask.any():
        hl_dt = tail.index[np.argmax(cross_mask.values)]
        hl_days = (hl_dt - peak_dt).total_seconds() / (3600 * 24)
        return {"half_life_dt": hl_dt, "half_life_days": hl_days}
    return {"half_life_dt": None, "half_life_days": None}


def compute_half_life_exponential(
    s: pd.Series,
    peak_dt: pd.Timestamp,
    baseline: float,
    min_points: int = 5,
    eps: float = 1e-6
) -> Dict[str, Optional[float]]:
    """
    Exponential-decay half-life via log-linear fit on the post-peak tail:
        y(t) ≈ baseline + A * exp(-t/τ)  => ln(y-baseline) = ln(A) - t/τ
    Fit ln(y-baseline) vs t (days) using numpy.polyfit to estimate τ,
    then t_1/2 = τ * ln(2).

    Parameters
    ----------
    s : pd.Series
        Time series with DatetimeIndex.
    peak_dt : pd.Timestamp
        Event peak datetime.
    baseline : float
        Baseline value.
    min_points : int
        Minimum tail points above baseline to attempt fit.
    eps : float
        Small value to ensure positivity for log.

    Returns
    -------
    dict
        {'half_life_days_exp': float or None}
    """
    tail = s[s.index > peak_dt]
    y = tail.values - baseline
    mask = y > eps
    if mask.sum() < min_points:
        return {"half_life_days_exp": None}

    y_pos = y[mask]
    t_days = (tail.index[mask] - peak_dt).total_seconds() / (3600 * 24)

    # ln(y) = intercept + slope * t, with slope ≈ -1/τ
    ln_y = np.log(y_pos)
    slope, intercept = np.polyfit(t_days, ln_y, 1)
    if slope >= 0:
        # Non-decaying or ill-conditioned
        return {"half_life_days_exp": None}
    tau = -1.0 / slope
    half_life_days_exp = float(tau * np.log(2.0))
    return {"half_life_days_exp": half_life_days_exp}


def compute_event_metrics(
    df: pd.DataFrame,
    value_col: str,
    dt_col: str = "dt",
    window: str = "30D",
    min_periods: int = 7,
    k: float = 5.0,
    baseline_window: str = "30D",
    min_run_points: int = 1,
    consolidate_gap: Optional[str] = None,
    use_exponential_fallback: bool = True
) -> pd.DataFrame:
    """
    End-to-end computation: detect MAD-based events and compute baseline,
    amplitude, and half-life (threshold method + optional exponential fallback).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing dt and value_col.
    value_col : str
        Column to analyze (e.g., 'n_docs', 'dispersion').
    dt_col : str
        Datetime column name, default 'dt'.
    window : str
        Rolling window for MAD detection (e.g., '30D').
    min_periods : int
        Minimum points for rolling statistics.
    k : float
        MAD multiplier.
    baseline_window : str
        Window for baseline calculation.
    min_run_points : int
        Minimum length of above-threshold run to keep as event.
    consolidate_gap : Optional[str]
        Merge events separated by gaps smaller than this duration.
    use_exponential_fallback : bool
        If True, compute exponential-fit half-life when threshold half-life is missing.

    Returns
    -------
    pd.DataFrame
        Columns:
        ['event_id', 'onset_dt', 'peak_dt', 'end_dt',
         'baseline', 'peak_value', 'amplitude',
         'half_life_dt', 'half_life_days', 'half_life_days_exp']
    """
    dfx = df[[dt_col, value_col]].dropna().copy()
    dfx[dt_col] = pd.to_datetime(dfx[dt_col])
    s = dfx.set_index(dt_col)[value_col].sort_index()

    events = detect_events_mad(
        s, window=window, min_periods=min_periods, k=k,
        min_points=min_run_points, consolidate_gap=consolidate_gap
    )

    if events.empty:
        return pd.DataFrame(columns=[
            "event_id", "onset_dt", "peak_dt", "end_dt",
            "baseline", "peak_value", "amplitude",
            "half_life_dt", "half_life_days", "half_life_days_exp"
        ])

    rows = []
    for _, ev in events.iterrows():
        onset_dt = ev["onset_dt"]
        peak_dt = ev["peak_dt"]
        end_dt = ev["end_dt"]
        peak_value = float(ev["peak_value"])

        baseline = compute_event_baseline(
            s, onset_dt=onset_dt, baseline_window=baseline_window, min_periods=min_periods
        )
        amplitude = float(peak_value - baseline)

        hl = compute_half_life_threshold(s, peak_dt=peak_dt, baseline=baseline, amplitude=amplitude)
        half_life_dt = hl["half_life_dt"]
        half_life_days = hl["half_life_days"]

        half_life_days_exp = None
        if use_exponential_fallback and (half_life_dt is None):
            hl_exp = compute_half_life_exponential(s, peak_dt=peak_dt, baseline=baseline)
            half_life_days_exp = hl_exp["half_life_days_exp"]

        rows.append({
            "event_id": int(ev["event_id"]),
            "onset_dt": onset_dt,
            "peak_dt": peak_dt,
            "end_dt": end_dt,
            "baseline": baseline,
            "peak_value": peak_value,
            "amplitude": amplitude,
            "half_life_dt": half_life_dt,
            "half_life_days": half_life_days,
            "half_life_days_exp": half_life_days_exp
        })

    return pd.DataFrame(rows)
