from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def _trapezoid_score(value: float, good: Tuple[float, float], soft: Tuple[float, float], *, inverse: bool = False) -> float:
    """Map ``value`` to a [0, 1] score using trapezoid membership.

    Parameters
    ----------
    value : float
        Metric value.
    good : Tuple[float, float]
        Interval with score=1.
    soft : Tuple[float, float]
        Outer interval where score linearly decays to 0.
    inverse : bool, optional
        If ``True`` smaller values are better, by default False.
    """
    if pd.isna(value):
        return 0.0

    a, b = good
    c, d = soft

    if inverse:
        # Expect ``a`` and ``c`` to represent the lower bound (often zero)
        if value <= b:
            return 1.0
        if value <= d and d != b:
            return (d - value) / (d - b)
        return 0.0

    if value < c or value > d:
        return 0.0
    if a <= value <= b:
        return 1.0
    if value < a and a != c:
        return (value - c) / (a - c)
    if value > b and d != b:
        return (d - value) / (d - b)
    return 0.0


def compute_levels_metrics(
    df: pd.DataFrame,
    levels_state: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute aggregate and per-level metrics for levels quality.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC dataframe with a ``DatetimeIndex``.
    levels_state : pd.DataFrame
        Levels dataframe after break/update step. Missing columns are allowed
        and will result in ``NaN`` metrics where appropriate.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``metrics_row_df``: single-row dataframe with aggregate metrics.
        ``per_level_df``: per-level metrics for debugging/analysis.
    """

    metrics = {}
    n_bars = len(df)
    metrics["n_bars"] = int(n_bars)

    n_levels = len(levels_state)
    metrics["n_levels"] = int(n_levels)
    metrics["density_per_1000"] = (n_levels / n_bars * 1000) if n_bars else 0.0

    # Peak/trough counts
    peaks = int((levels_state["type"] == "peak").sum()) if "type" in levels_state else 0
    troughs = int((levels_state["type"] == "trough").sum()) if "type" in levels_state else 0
    metrics["peaks"] = peaks
    metrics["troughs"] = troughs
    metrics["peak_trough_ratio"] = float(peaks / max(1, troughs)) if n_levels else float("nan")

    # States
    n_broken = int((levels_state["state"] == "broken").sum()) if "state" in levels_state else 0
    metrics["n_broken"] = n_broken
    metrics["n_intact"] = int(n_levels - n_broken)
    metrics["share_broken"] = n_broken / max(1, n_levels)

    # Weak proportionality share
    if "weak_prop" in levels_state:
        metrics["weak_prop_share"] = float(levels_state["weak_prop"].mean())
    else:
        metrics["weak_prop_share"] = float("nan")

    # Time to break (bars)
    ttb_series = pd.Series(dtype="float64")
    if {"break_idx", "end_idx"}.issubset(levels_state.columns):
        mask = (
            (levels_state["state"] == "broken")
            if "state" in levels_state
            else pd.Series([True] * len(levels_state), index=levels_state.index)
        )
        mask &= levels_state["break_idx"] >= 0
        ttb_series = levels_state.loc[mask, "break_idx"].astype(float) - levels_state.loc[
            mask, "end_idx"
        ].astype(float)
        metrics["ttb_median"] = float(ttb_series.median()) if not ttb_series.empty else float("nan")
        metrics["ttb_mean"] = float(ttb_series.mean()) if not ttb_series.empty else float("nan")
        metrics["ttb_p25"] = float(ttb_series.quantile(0.25)) if not ttb_series.empty else float("nan")
        metrics["ttb_p75"] = float(ttb_series.quantile(0.75)) if not ttb_series.empty else float("nan")
    else:
        metrics["ttb_median"] = metrics["ttb_mean"] = metrics["ttb_p25"] = metrics["ttb_p75"] = float("nan")

    # Time to first touch
    if {"first_touch_time", "end_idx"}.issubset(levels_state.columns):
        mask = levels_state["first_touch_time"].notna()
        if mask.any():
            idx_pos = df.index.get_indexer(levels_state.loc[mask, "first_touch_time"])
            tft_series = (
                pd.Series(idx_pos, index=levels_state.index[mask]).astype(float)
                - levels_state.loc[mask, "end_idx"].astype(float)
            )
            metrics["tft_median"] = float(tft_series.median()) if not tft_series.empty else float("nan")
        else:
            tft_series = pd.Series(dtype="float64")
            metrics["tft_median"] = float("nan")
        metrics["share_touched"] = mask.sum() / max(1, n_levels)
    else:
        tft_series = pd.Series(dtype="float64")
        metrics["tft_median"] = float("nan")
        metrics["share_touched"] = float("nan")

    # Update stats
    if "update_count" in levels_state:
        update_series = levels_state["update_count"].astype(float)
        metrics["update_mean"] = float(update_series.mean()) if n_levels else float("nan")
        metrics["update_p75"] = float(update_series.quantile(0.75)) if n_levels else float("nan")
    else:
        metrics["update_mean"] = metrics["update_p75"] = float("nan")

    # Break strength
    strength_series = pd.Series(dtype="float64")
    required_cols = {"confirm_threshold", "price", "break_idx", "type", "state"}
    if required_cols.issubset(levels_state.columns):
        mask = (levels_state["state"] == "broken") & (levels_state["break_idx"] >= 0)
        if mask.any():
            close_at_break = df["close"].iloc[levels_state.loc[mask, "break_idx"].astype(int).values]
            price_at_break = levels_state.loc[mask, "price"].astype(float).values
            thresholds = levels_state.loc[mask, "confirm_threshold"].astype(float).values
            types = levels_state.loc[mask, "type"].values
            vals = []
            for close_val, price0, thr, tp in zip(
                close_at_break, price_at_break, thresholds, types
            ):
                thr = max(thr, 1e-12)
                if tp == "peak":
                    margin = close_val - (price0 + thr)
                else:
                    margin = (price0 - thr) - close_val
                vals.append(max(0.0, margin) / thr)
            strength_series = pd.Series(vals, index=levels_state.index[mask])
        metrics["break_strength_median"] = (
            float(strength_series.median()) if not strength_series.empty else float("nan")
        )
    else:
        metrics["break_strength_median"] = float("nan")

    metrics_row_df = pd.DataFrame([metrics])

    # Per-level metrics
    cols = [c for c in ["id", "time", "type", "state", "end_idx", "break_idx", "first_touch_time", "update_count", "price", "confirm_threshold", "weak_prop"] if c in levels_state.columns]
    per_level_df = levels_state[cols].copy() if cols else pd.DataFrame(index=levels_state.index)
    if {"break_idx", "end_idx"}.issubset(levels_state.columns):
        per_level_df["time_to_break_bars"] = levels_state["break_idx"] - levels_state["end_idx"]
    else:
        per_level_df["time_to_break_bars"] = pd.Series(float("nan"), index=levels_state.index)
    if {"first_touch_time", "end_idx"}.issubset(levels_state.columns):
        mask = levels_state["first_touch_time"].notna()
        if mask.any():
            idx_pos = df.index.get_indexer(levels_state.loc[mask, "first_touch_time"])
            tft_vals = (
                pd.Series(idx_pos, index=levels_state.index[mask]) - levels_state.loc[mask, "end_idx"]
            )
            per_level_df["time_to_first_touch"] = pd.Series(float("nan"), index=levels_state.index)
            per_level_df.loc[mask, "time_to_first_touch"] = tft_vals
        else:
            per_level_df["time_to_first_touch"] = pd.Series(float("nan"), index=levels_state.index)
    else:
        per_level_df["time_to_first_touch"] = pd.Series(float("nan"), index=levels_state.index)
    if not strength_series.empty:
        per_level_df["break_strength_norm"] = strength_series
    else:
        per_level_df["break_strength_norm"] = pd.Series(float("nan"), index=levels_state.index)

    return metrics_row_df, per_level_df


def score_levels(metrics: pd.Series, tf: str, eval_cfg: Dict) -> Dict[str, float]:
    """Compute score breakdown and total score for levels metrics."""
    cfg = eval_cfg.get("levels_scoring", {})
    weights = cfg.get("weights", {})
    targets = cfg.get("targets", {}).get(tf, {})

    scores = {}
    density_cfg = targets.get("density", {})
    scores["score_density"] = _trapezoid_score(
        metrics.get("density_per_1000"),
        tuple(density_cfg.get("good", [0.0, 0.0])),
        tuple(density_cfg.get("soft", [0.0, 0.0])),
    )

    broken_cfg = targets.get("share_broken", {})
    scores["score_share_broken"] = _trapezoid_score(
        metrics.get("share_broken"),
        tuple(broken_cfg.get("good", [0.0, 0.0])),
        tuple(broken_cfg.get("soft", [0.0, 0.0])),
    )

    weak_cfg = targets.get("weak_prop_share", {})
    scores["score_weak_prop"] = _trapezoid_score(
        metrics.get("weak_prop_share"),
        tuple(weak_cfg.get("good", [0.0, 0.0])),
        tuple(weak_cfg.get("soft", [0.0, 0.0])),
        inverse=True,
    )

    ttb_cfg = targets.get("ttb_median", {})
    scores["score_ttb"] = _trapezoid_score(
        metrics.get("ttb_median"),
        tuple(ttb_cfg.get("good", [0.0, 0.0])),
        tuple(ttb_cfg.get("soft", [0.0, 0.0])),
    )

    balance_cfg = targets.get("balance_peak_trough", {})
    scores["score_balance"] = _trapezoid_score(
        metrics.get("peak_trough_ratio"),
        tuple(balance_cfg.get("good", [0.0, 0.0])),
        tuple(balance_cfg.get("soft", [0.0, 0.0])),
    )

    total = (
        weights.get("density", 0.0) * scores["score_density"]
        + weights.get("share_broken", 0.0) * scores["score_share_broken"]
        + weights.get("weak_prop", 0.0) * scores["score_weak_prop"]
        + weights.get("ttb", 0.0) * scores["score_ttb"]
        + weights.get("balance", 0.0) * scores["score_balance"]
    )
    scores["score_total"] = float(total)
    return scores
