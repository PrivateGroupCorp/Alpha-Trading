from __future__ import annotations

"""Utilities to build price swings from processed levels."""

from dataclasses import dataclass, asdict
from typing import List, Literal

import pandas as pd

from alpha.core.indicators import atr

SwingType = Literal["peak", "trough"]


@dataclass
class SwingsCfg:
    """Configuration for swing construction."""

    use_only_strong_swings: bool = True
    swing_merge_atr_mult: float = 0.10
    min_gap_bars: int = 1
    min_price_delta_atr_mult: float = 0.05
    keep_latest_on_tie: bool = True
    atr_window: int = 14


def _ensure_level_ids(levels: pd.DataFrame) -> pd.DataFrame:
    if "level_id" not in levels.columns:
        levels = levels.copy()
        levels["level_id"] = range(len(levels))
    return levels


def build_swings(df: pd.DataFrame, levels_df: pd.DataFrame, cfg: SwingsCfg) -> pd.DataFrame:
    """Build swings from level dataframe following provided configuration."""

    levels_df = _ensure_level_ids(levels_df)
    levels_df = levels_df.sort_values("end_idx").reset_index(drop=True)

    # initial filtering
    if cfg.use_only_strong_swings and "weak_prop" in levels_df.columns:
        levels_df = levels_df[~levels_df["weak_prop"].astype(bool)].copy()

    cols = [c for c in ["time", "end_idx", "type", "price", "weak_prop", "level_id"] if c in levels_df.columns]
    levels = levels_df[cols].copy()

    atr_series = atr(df, window=cfg.atr_window)
    eps = 1e-12

    swings: List[dict] = []
    merge_same = 0

    for _, row in levels.iterrows():
        swing = {
            "time": row["time"],
            "idx": int(row["end_idx"]),
            "type": row["type"],
            "price": float(row["price"]),
            "src_level_ids": [row["level_id"]],
            "merged_count": 1,
            "weak_any": bool(row.get("weak_prop", False)),
        }
        if not swings:
            swings.append(swing)
            continue
        last = swings[-1]
        if swing["type"] == last["type"]:
            merge_same += 1
            if swing["type"] == "peak":
                better = swing["price"] > last["price"] or (
                    abs(swing["price"] - last["price"]) <= eps and cfg.keep_latest_on_tie
                )
            else:  # trough
                better = swing["price"] < last["price"] or (
                    abs(swing["price"] - last["price"]) <= eps and cfg.keep_latest_on_tie
                )
            if better:
                swing["src_level_ids"] = last["src_level_ids"] + swing["src_level_ids"]
                swing["merged_count"] = last["merged_count"] + swing["merged_count"]
                swing["weak_any"] = last["weak_any"] or swing["weak_any"]
                swings[-1] = swing
            else:
                last["src_level_ids"].extend(swing["src_level_ids"])
                last["merged_count"] += swing["merged_count"]
                last["weak_any"] = last["weak_any"] or swing["weak_any"]
        else:
            swings.append(swing)

    # merge nearby swings
    merge_near = 0
    i = 1
    while i < len(swings):
        prev = swings[i - 1]
        cur = swings[i]
        gap = cur["idx"] - prev["idx"]
        price_delta = abs(cur["price"] - prev["price"])
        atr_ref = float(atr_series.iloc[cur["idx"]]) if len(atr_series) > cur["idx"] else 0.0
        if atr_ref < eps:
            atr_ref = eps
        cond = False
        if price_delta <= atr_ref * cfg.swing_merge_atr_mult + eps:
            cond = True
        if gap < cfg.min_gap_bars:
            cond = True
        if price_delta <= atr_ref * cfg.min_price_delta_atr_mult + eps:
            cond = True
        if cond:
            merge_near += 1
            if cur["type"] == "peak":
                better = cur["price"] > prev["price"] or (
                    abs(cur["price"] - prev["price"]) <= eps and cfg.keep_latest_on_tie
                )
            else:
                better = cur["price"] < prev["price"] or (
                    abs(cur["price"] - prev["price"]) <= eps and cfg.keep_latest_on_tie
                )
            if better:
                cur["src_level_ids"] = prev["src_level_ids"] + cur["src_level_ids"]
                cur["merged_count"] = prev["merged_count"] + cur["merged_count"]
                cur["weak_any"] = prev["weak_any"] or cur["weak_any"]
                swings[i - 1] = cur
                swings.pop(i)
            else:
                prev["src_level_ids"].extend(cur["src_level_ids"])
                prev["merged_count"] += cur["merged_count"]
                prev["weak_any"] = prev["weak_any"] or cur["weak_any"]
                swings.pop(i)
        else:
            i += 1

    # compute helper fields
    for k, sw in enumerate(swings):
        sw["swing_id"] = k
        sw["src_level_ids"] = ",".join(map(str, sw["src_level_ids"]))
        sw["gap_from_prev"] = (
            sw["idx"] - swings[k - 1]["idx"] if k > 0 else pd.NA
        )
        sw["leg_from_prev"] = (
            abs(sw["price"] - swings[k - 1]["price"]) if k > 0 else float("nan")
        )
        sw["leg_to_next"] = (
            abs(swings[k + 1]["price"] - sw["price"]) if k < len(swings) - 1 else float("nan")
        )
        sw["atr_at_idx"] = (
            float(atr_series.iloc[sw["idx"]]) if len(atr_series) > sw["idx"] else float("nan")
        )

    out_cols = [
        "swing_id",
        "time",
        "idx",
        "type",
        "price",
        "src_level_ids",
        "merged_count",
        "weak_any",
        "gap_from_prev",
        "leg_from_prev",
        "leg_to_next",
        "atr_at_idx",
    ]
    out_df = pd.DataFrame(swings, columns=out_cols)
    out_df.attrs["merge_same_type_count"] = merge_same
    out_df.attrs["merge_nearby_count"] = merge_near
    return out_df


def summarize_swings(swings_df: pd.DataFrame, levels_df: pd.DataFrame, cfg: SwingsCfg) -> dict:
    """Produce summary statistics for swings."""

    n_levels_in = int(len(levels_df))
    n_levels_used = int(swings_df["merged_count"].sum()) if len(swings_df) else 0
    n_swings = int(len(swings_df))
    weak_ratio_in = (
        float(levels_df.get("weak_prop", pd.Series(dtype="float64")).mean())
        if ("weak_prop" in levels_df.columns and len(levels_df))
        else 0.0
    )
    weak_ratio_used = float(swings_df["weak_any"].mean()) if len(swings_df) else 0.0
    median_leg = (
        float(swings_df["leg_from_prev"].median(skipna=True)) if len(swings_df) else 0.0
    )

    summary = {
        "n_levels_in": n_levels_in,
        "n_levels_used": n_levels_used,
        "n_swings": n_swings,
        "merge_same_type_count": int(swings_df.attrs.get("merge_same_type_count", 0)),
        "merge_nearby_count": int(swings_df.attrs.get("merge_nearby_count", 0)),
        "weak_ratio_in": weak_ratio_in,
        "weak_ratio_used": weak_ratio_used,
        "median_leg_from_prev": median_leg,
        "params": asdict(cfg),
    }
    return summary
