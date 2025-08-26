"""Liquidity sweep / grab detection utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

from alpha.core.indicators import atr


Scope = Literal["eq_cluster", "asia_high", "asia_low"]
Side = Literal["up", "down"]


@dataclass
class SweepCfg:
    """Configuration for sweep detection."""

    confirm_mode: str = "body"  # "body" | "wick"
    reclaim_mode: str = "body_back"  # "body_back" | "close_back_n_bars" | "bos_opposite_link"
    reclaim_n_bars: int = 3
    min_pen_atr_mult: float = 0.10
    max_pen_atr_mult: float = 1.00
    max_duration_bars: int = 10
    link_events_window_bars: int = 20
    atr_window: int = 14
    pip_size: float = 0.0001
    tick_size: float = 0.0
    quality_weights: dict | None = None
    quality_grades: dict | None = None


def _default_quality_weights() -> dict:
    return {
        "penetration_norm": 0.30,
        "fast_reclaim": 0.35,
        "context_eq": 0.20,
        "asia_alignment": 0.15,
    }


def _default_quality_grades() -> dict:
    return {"A": [0.75, 1.00], "B": [0.55, 0.75], "C": [0.00, 0.55]}


def _compute_pen_norm(pen_atr: float, cfg: SweepCfg) -> float:
    min_pen = cfg.min_pen_atr_mult
    max_pen = cfg.max_pen_atr_mult
    mid = (min_pen + max_pen) / 2
    if pen_atr <= min_pen:
        return 0.0
    if pen_atr >= max_pen:
        return 0.0
    if pen_atr <= mid:
        denom = max(mid - min_pen, 1e-12)
        return (pen_atr - min_pen) / denom
    denom = max(max_pen - mid, 1e-12)
    return max(0.0, 1 - (pen_atr - mid) / denom)


def _quality_grade(score: float, grades: dict) -> str:
    for g, (lo, hi) in grades.items():
        if lo <= score <= hi:
            return g
    return "C"


def detect_sweeps(
    df: pd.DataFrame,
    eq_clusters: pd.DataFrame,
    asia_daily: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    cfg: SweepCfg,
) -> pd.DataFrame:
    """Detect liquidity sweeps on EQ clusters and Asia range edges."""

    atr_series = atr(df, window=cfg.atr_window)
    eps = 1e-12

    edges: list[dict] = []

    if eq_clusters is not None and not eq_clusters.empty:
        for row in eq_clusters.itertuples():
            edges.append(
                {
                    "scope": "eq_cluster",
                    "side": "up" if getattr(row, "side", "eqh") == "eqh" else "down",
                    "price": float(getattr(row, "price_center")),
                    "meta": {
                        "cluster_id": getattr(row, "cluster_id", None),
                        "cluster_score": float(getattr(row, "score", np.nan)),
                        "cluster_width_atr": float(getattr(row, "width_atr", np.nan)),
                        "first_time": getattr(row, "first_time", pd.NaT),
                        "last_time": getattr(row, "last_time", pd.NaT),
                    },
                }
            )

    if asia_daily is not None and not asia_daily.empty:
        for row in asia_daily.itertuples():
            edges.append(
                {
                    "scope": "asia_high",
                    "side": "up",
                    "price": float(getattr(row, "asia_high")),
                    "meta": {
                        "date": getattr(row, "date", None),
                        "post_first_break_dir": getattr(
                            row, "post_first_break_dir", "none"
                        ),
                        "end_ts": getattr(row, "end_ts", pd.NaT),
                    },
                }
            )
            edges.append(
                {
                    "scope": "asia_low",
                    "side": "down",
                    "price": float(getattr(row, "asia_low")),
                    "meta": {
                        "date": getattr(row, "date", None),
                        "post_first_break_dir": getattr(
                            row, "post_first_break_dir", "none"
                        ),
                        "end_ts": getattr(row, "end_ts", pd.NaT),
                    },
                }
            )

    records: list[dict] = []
    sweep_id = 0
    weights = cfg.quality_weights or _default_quality_weights()
    grades = cfg.quality_grades or _default_quality_grades()

    for edge in edges:
        price = edge["price"]
        side = edge["side"]
        scope = edge["scope"]
        meta = edge["meta"]

        start_ts = meta.get("last_time") if scope == "eq_cluster" else meta.get("end_ts")
        if pd.notna(start_ts):
            start_idx = int(df.index.searchsorted(pd.to_datetime(start_ts))) + 1
        else:
            start_idx = 0

        for j in range(start_idx, len(df)):
            bar = df.iloc[j]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            if cfg.confirm_mode == "body":
                pen = close > price if side == "up" else close < price
            else:
                pen = high > price if side == "up" else low < price
            if not pen:
                continue

            extreme = high if side == "up" else low
            pen_depth = abs(extreme - price)
            atr_j = float(atr_series.iloc[j]) if j < len(atr_series) else 0.0
            pen_atr = pen_depth / max(atr_j, eps)
            if pen_atr < cfg.min_pen_atr_mult or pen_atr > cfg.max_pen_atr_mult:
                continue

            linked_event_id = np.nan
            linked_event = np.nan
            linked_direction = np.nan
            bars_to_event = np.nan
            if events_df is not None and "idx" in events_df.columns:
                diff_all = events_df["idx"] - j
                mask = diff_all.abs() <= cfg.link_events_window_bars
                if mask.any():
                    idx_min = (diff_all[mask].abs()).idxmin()
                    ev = events_df.loc[idx_min]
                    linked_event_id = int(ev.get("event_id", ev.get("idx")))
                    linked_event = ev.get("event")
                    linked_direction = ev.get("direction")
                    bars_to_event = int(diff_all.loc[idx_min])

            reclaim_ok = False
            r_idx = np.nan
            max_idx = min(j + cfg.max_duration_bars, len(df) - 1)
            if cfg.reclaim_mode == "body_back":
                for k in range(j + 1, max_idx + 1):
                    close_k = float(df.iloc[k]["close"])
                    if (side == "up" and close_k <= price) or (
                        side == "down" and close_k >= price
                    ):
                        reclaim_ok = True
                        r_idx = k
                        break
            elif cfg.reclaim_mode == "close_back_n_bars":
                max_idx = min(j + cfg.reclaim_n_bars, len(df) - 1)
                for k in range(j + 1, max_idx + 1):
                    close_k = float(df.iloc[k]["close"])
                    if (side == "up" and close_k <= price) or (
                        side == "down" and close_k >= price
                    ):
                        reclaim_ok = True
                        r_idx = k
                        break
            elif cfg.reclaim_mode == "bos_opposite_link":
                if events_df is not None and "idx" in events_df.columns:
                    opp_dir = "down" if side == "up" else "up"
                    cond = events_df[events_df["direction"] == opp_dir]
                    diff2 = cond["idx"] - j
                    mask2 = diff2.abs() <= cfg.link_events_window_bars
                    if mask2.any():
                        idx_min2 = (diff2[mask2].abs()).idxmin()
                        ev2 = cond.loc[idx_min2]
                        reclaim_ok = True
                        r_idx = int(ev2["idx"])
                        linked_event_id = int(ev2.get("event_id", ev2.get("idx")))
                        linked_event = ev2.get("event")
                        linked_direction = ev2.get("direction")
                        bars_to_event = int(diff2.loc[idx_min2])

            duration = int(r_idx - j) if reclaim_ok and not np.isnan(r_idx) else np.nan

            q_pen = _compute_pen_norm(pen_atr, cfg)
            q_fast = (
                max(0.0, 1 - duration / cfg.max_duration_bars)
                if reclaim_ok and not np.isnan(duration)
                else 0.0
            )
            if scope == "eq_cluster":
                q_ctx = float(np.clip(meta.get("cluster_score", 0.0), 0.0, 1.0))
            else:
                q_ctx = 0.5
            if scope.startswith("asia"):
                post_dir = meta.get("post_first_break_dir", "none")
                if post_dir == side:
                    q_asia = 1.0
                elif post_dir == "none":
                    q_asia = 0.5
                else:
                    q_asia = 0.0
            else:
                q_asia = 0.5

            quality_score = (
                weights.get("penetration_norm", 0) * q_pen
                + weights.get("fast_reclaim", 0) * q_fast
                + weights.get("context_eq", 0) * q_ctx
                + weights.get("asia_alignment", 0) * q_asia
            )
            quality_score = float(np.clip(quality_score, 0.0, 1.0))
            grade = _quality_grade(quality_score, grades)

            record = {
                "sweep_id": sweep_id,
                "scope": scope,
                "side": side,
                "pen_idx": j,
                "pen_time": df.index[j],
                "reclaim_idx": r_idx,
                "reclaim_time": df.index[int(r_idx)] if reclaim_ok and not np.isnan(r_idx) else pd.NaT,
                "duration_bars": duration,
                "pen_depth": pen_depth,
                "pen_depth_pips": pen_depth / cfg.pip_size if cfg.pip_size else np.nan,
                "pen_depth_atr": pen_atr,
                "reclaim_mode": cfg.reclaim_mode,
                "reclaim_ok": bool(reclaim_ok),
                "reclaim_threshold_note": "",
                "linked_event_id": linked_event_id,
                "linked_event": linked_event,
                "linked_direction": linked_direction,
                "bars_to_event": bars_to_event,
                "quality_score": quality_score,
                "quality_grade": grade,
                "q_penetration_norm": q_pen,
                "q_fast_reclaim": q_fast,
                "q_context_eq": q_ctx,
                "q_asia_alignment": q_asia,
            }

            if scope == "eq_cluster":
                record.update(
                    {
                        "cluster_id": meta.get("cluster_id"),
                        "cluster_price_center": price,
                        "cluster_width_atr": meta.get("cluster_width_atr"),
                        "cluster_score": meta.get("cluster_score"),
                    }
                )
            else:
                record.update({"date": meta.get("date"), "edge_price": price})

            records.append(record)
            sweep_id += 1
            break

    if not records:
        return pd.DataFrame(
            columns=[
                "sweep_id",
                "scope",
                "side",
                "pen_idx",
                "pen_time",
                "reclaim_idx",
                "reclaim_time",
                "duration_bars",
                "pen_depth",
                "pen_depth_pips",
                "pen_depth_atr",
                "reclaim_mode",
                "reclaim_ok",
                "reclaim_threshold_note",
                "linked_event_id",
                "linked_event",
                "linked_direction",
                "bars_to_event",
                "quality_score",
                "quality_grade",
                "q_penetration_norm",
                "q_fast_reclaim",
                "q_context_eq",
                "q_asia_alignment",
                "cluster_id",
                "cluster_price_center",
                "cluster_width_atr",
                "cluster_score",
                "date",
                "edge_price",
            ]
        )

    out = pd.DataFrame(records)
    return out


def summarize_sweeps(sweeps_df: pd.DataFrame, cfg: SweepCfg) -> dict:
    """Summarize sweep dataframe into basic statistics."""

    n = int(len(sweeps_df))
    share_eq = float((sweeps_df["scope"] == "eq_cluster").mean()) if n else 0.0
    share_asia = float((sweeps_df["scope"].str.startswith("asia").mean())) if n else 0.0
    median_pen_pips = (
        float(sweeps_df["pen_depth_pips"].median()) if n else float("nan")
    )
    median_dur = (
        float(sweeps_df["duration_bars"].median()) if n else float("nan")
    )
    grade_counts = (
        sweeps_df["quality_grade"].value_counts().to_dict() if n else {}
    )
    link_rate = (
        float(sweeps_df["linked_event_id"].notna().mean()) if n else 0.0
    )

    summary = {
        "n_sweeps": n,
        "share_scope_eq": share_eq,
        "share_scope_asia": share_asia,
        "median_pen_depth_pips": median_pen_pips,
        "median_duration_bars": median_dur,
        "grade_counts": grade_counts,
        "link_rate_to_events": link_rate,
        "params": asdict(cfg),
    }
    return summary

