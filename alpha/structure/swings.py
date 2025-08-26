from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas import Timestamp

from alpha.core.indicators import atr

SwingType = Literal["peak", "trough"]


@dataclass
class SwingsCfg:
    use_only_strong_swings: bool = True
    swing_merge_atr_mult: float = 0.10
    min_gap_bars: int = 1
    min_price_delta_atr_mult: float = 0.05
    keep_latest_on_tie: bool = True
    atr_window: int = 14


def _prepare_levels(levels_df: pd.DataFrame, cfg: SwingsCfg) -> pd.DataFrame:
    lv = levels_df.copy()
    if "time" in lv.columns:
        lv["time"] = pd.to_datetime(lv["time"], utc=True, errors="coerce")
    sort_col = "end_idx" if "end_idx" in lv.columns else "time"
    lv = lv.sort_values(sort_col).reset_index(drop=True)
    if "level_id" not in lv.columns:
        lv["level_id"] = range(len(lv))
    cols = [c for c in ["time", "end_idx", "type", "price", "weak_prop", "level_id"] if c in lv.columns]
    lv = lv[cols]
    if cfg.use_only_strong_swings and "weak_prop" in lv.columns:
        lv = lv[lv["weak_prop"] != True].reset_index(drop=True)
    return lv


def build_swings(df: pd.DataFrame, levels_df: pd.DataFrame, cfg: SwingsCfg) -> pd.DataFrame:
    levels = _prepare_levels(levels_df, cfg)
    atr_series = atr(df, window=cfg.atr_window)
    eps = 1e-12

    swings: list[dict] = []
    merge_same = 0
    for row in levels.itertuples():
        swing = {
            "time": row.time if isinstance(row.time, Timestamp) else pd.to_datetime(row.time, utc=True),
            "idx": int(getattr(row, "end_idx", getattr(row, "idx", 0))),
            "type": row.type,
            "price": float(row.price),
            "src_level_ids": [int(row.level_id)],
            "merged_count": 1,
            "weak_any": bool(getattr(row, "weak_prop", False)),
        }
        if not swings:
            swings.append(swing)
            continue
        last = swings[-1]
        if swing["type"] == last["type"]:
            merge_same += 1
            if swing["type"] == "peak":
                take_new = swing["price"] > last["price"] or (
                    swing["price"] == last["price"] and cfg.keep_latest_on_tie
                )
            else:
                take_new = swing["price"] < last["price"] or (
                    swing["price"] == last["price"] and cfg.keep_latest_on_tie
                )
            last["src_level_ids"].extend(swing["src_level_ids"])
            last["merged_count"] += 1
            last["weak_any"] = last["weak_any"] or swing["weak_any"]
            if take_new:
                last.update({k: swing[k] for k in ["time", "idx", "price"]})
        else:
            swings.append(swing)

    merge_near = 0
    i = 1
    while i < len(swings):
        prev = swings[i - 1]
        cur = swings[i]
        price_delta = abs(cur["price"] - prev["price"])
        atr_ref = float(atr_series.iloc[cur["idx"]]) if cur["idx"] < len(atr_series) else 0.0
        atr_ref = max(atr_ref, eps)
        gap_bars = cur["idx"] - prev["idx"]
        if (
            price_delta <= atr_ref * cfg.swing_merge_atr_mult
            or gap_bars < cfg.min_gap_bars
            or price_delta <= atr_ref * cfg.min_price_delta_atr_mult
        ):
            merge_near += 1
            if cur["type"] == "peak":
                keep_cur = cur["price"] >= prev["price"]
                if cur["price"] == prev["price"]:
                    keep_cur = cfg.keep_latest_on_tie
            else:
                keep_cur = cur["price"] <= prev["price"]
                if cur["price"] == prev["price"]:
                    keep_cur = cfg.keep_latest_on_tie
            if keep_cur:
                cur["src_level_ids"] = prev["src_level_ids"] + cur["src_level_ids"]
                cur["merged_count"] += prev["merged_count"]
                cur["weak_any"] = cur["weak_any"] or prev["weak_any"]
                if cur["type"] == "peak" and prev["price"] > cur["price"]:
                    cur.update({k: prev[k] for k in ["time", "idx", "price"]})
                if cur["type"] == "trough" and prev["price"] < cur["price"]:
                    cur.update({k: prev[k] for k in ["time", "idx", "price"]})
                del swings[i - 1]
            else:
                prev["src_level_ids"].extend(cur["src_level_ids"])
                prev["merged_count"] += cur["merged_count"]
                prev["weak_any"] = prev["weak_any"] or cur["weak_any"]
                if cur["type"] == "peak" and cur["price"] > prev["price"]:
                    prev.update({k: cur[k] for k in ["time", "idx", "price"]})
                if cur["type"] == "trough" and cur["price"] < prev["price"]:
                    prev.update({k: cur[k] for k in ["time", "idx", "price"]})
                del swings[i]
            i = max(i - 1, 1)
        else:
            i += 1

    # Final pass to enforce alternation
    final_sw: list[dict] = []
    for sw in swings:
        if final_sw and sw["type"] == final_sw[-1]["type"]:
            merge_near += 1
            last = final_sw[-1]
            if sw["type"] == "peak":
                take_new = sw["price"] > last["price"] or (
                    sw["price"] == last["price"] and cfg.keep_latest_on_tie
                )
            else:
                take_new = sw["price"] < last["price"] or (
                    sw["price"] == last["price"] and cfg.keep_latest_on_tie
                )
            last["src_level_ids"].extend(sw["src_level_ids"])
            last["merged_count"] += sw["merged_count"]
            last["weak_any"] = last["weak_any"] or sw["weak_any"]
            if take_new:
                last.update({k: sw[k] for k in ["time", "idx", "price"]})
        else:
            final_sw.append(sw)
    swings = final_sw

    # Build DataFrame with helper fields
    out = pd.DataFrame(swings)
    if out.empty:
        out["swing_id"] = []
        out["leg_from_prev"] = []
        out["leg_to_next"] = []
        out["atr_at_idx"] = []
        out["min_gap_bars"] = []
        out.attrs["merge_same_type_count"] = merge_same
        out.attrs["merge_nearby_count"] = merge_near
        return out

    out["swing_id"] = range(len(out))
    leg_from_prev = [float("nan")]
    gap_from_prev = [0]
    for i in range(1, len(out)):
        leg_from_prev.append(abs(out.loc[i, "price"] - out.loc[i - 1, "price"]))
        gap_from_prev.append(int(out.loc[i, "idx"] - out.loc[i - 1, "idx"]))
    leg_to_next = [abs(out.loc[i + 1, "price"] - out.loc[i, "price"]) for i in range(len(out) - 1)]
    leg_to_next.append(float("nan"))
    atr_vals = [float(atr_series.iloc[int(idx)]) if int(idx) < len(atr_series) else float("nan") for idx in out["idx"]]

    out["leg_from_prev"] = leg_from_prev
    out["leg_to_next"] = leg_to_next
    out["atr_at_idx"] = atr_vals
    out["min_gap_bars"] = gap_from_prev

    out = out[
        [
            "swing_id",
            "time",
            "idx",
            "type",
            "price",
            "src_level_ids",
            "merged_count",
            "weak_any",
            "min_gap_bars",
            "leg_from_prev",
            "leg_to_next",
            "atr_at_idx",
        ]
    ]

    out.attrs["merge_same_type_count"] = merge_same
    out.attrs["merge_nearby_count"] = merge_near
    return out


def summarize_swings(swings_df: pd.DataFrame, levels_df: pd.DataFrame, cfg: SwingsCfg) -> dict:
    n_levels_in = int(len(levels_df))
    n_swings = int(len(swings_df))
    n_levels_used = int(swings_df["merged_count"].sum()) if n_swings else 0
    merge_same = int(swings_df.attrs.get("merge_same_type_count", 0))
    merge_near = int(swings_df.attrs.get("merge_nearby_count", 0))

    weak_ratio_in = 0.0
    weak_ratio_used = 0.0
    if "weak_prop" in levels_df.columns and n_levels_in:
        weak_ratio_in = float(levels_df["weak_prop"].mean())
        if n_levels_used:
            if "level_id" not in levels_df.columns:
                levels_df = levels_df.copy()
                levels_df["level_id"] = range(len(levels_df))
            weak_map = levels_df.set_index("level_id")["weak_prop"].to_dict()
            used_ids: list[int] = []
            for ids in swings_df["src_level_ids"]:
                used_ids.extend(ids)
            weak_vals = [weak_map.get(i, False) for i in used_ids]
            weak_ratio_used = float(pd.Series(weak_vals).mean()) if used_ids else 0.0

    median_leg = (
        float(swings_df["leg_from_prev"].median(skipna=True)) if n_swings else 0.0
    )

    return {
        "n_levels_in": n_levels_in,
        "n_levels_used": n_levels_used,
        "n_swings": n_swings,
        "merge_same_type_count": merge_same,
        "merge_nearby_count": merge_near,
        "weak_ratio_in": weak_ratio_in,
        "weak_ratio_used": weak_ratio_used,
        "median_leg_from_prev": median_leg,
        "params": {
            "use_only_strong_swings": cfg.use_only_strong_swings,
            "swing_merge_atr_mult": cfg.swing_merge_atr_mult,
            "min_gap_bars": cfg.min_gap_bars,
            "min_price_delta_atr_mult": cfg.min_price_delta_atr_mult,
            "keep_latest_on_tie": cfg.keep_latest_on_tie,
            "atr_window": cfg.atr_window,
        },
    }
