from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import math
import numpy as np
import pandas as pd

from alpha.core.indicators import atr

Kind = Literal["demand", "supply", "breaker", "flip"]



@dataclass
class PoiCfg:
    use_only_valid_events: bool = True
    # ob
    ob_lookback_bars: int = 15
    ob_max_anchor_gap_bars: int = 8
    ob_zone_padding_atr_mult: float = 0.05
    ob_body_ratio_min: float = 0.25
    ob_prefer_full_wick: bool = True
    fvg_detect: bool = True
    # merge
    merge_overlap: bool = True
    merge_gap_atr_mult: float = 0.15
    min_width_atr: float = 0.02
    max_width_atr: float = 0.60
    min_age_bars: int = 2
    # score
    score_weights: dict | None = None
    grade_map: dict | None = None
    # format
    atr_window: int = 14
    pip_size: float = 0.0001
    tick_size: float = 0.0


def _default_weights() -> dict:
    return {
        "structure": 0.35,
        "sweep": 0.25,
        "liquidity": 0.15,
        "trend": 0.15,
        "tech": 0.10,
    }


def _default_grades() -> dict:
    return {"A": [0.75, 1.0], "B": [0.55, 0.75], "C": [0.0, 0.55]}


def _grade(score: float, grades: dict) -> str:
    for g, (lo, hi) in grades.items():
        if lo <= score <= hi:
            return g
    return "C"


def _round_price(x: float, tick: float, up: bool) -> float:
    if tick <= 0:
        return x
    if up:
        return math.ceil(x / tick) * tick
    return math.floor(x / tick) * tick


def build_poi_zones(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    events: pd.DataFrame,
    sweeps: Optional[pd.DataFrame],
    eq_clusters: Optional[pd.DataFrame],
    asia_daily: Optional[pd.DataFrame],
    trend_timeline: Optional[pd.DataFrame],
    cfg: PoiCfg,
) -> pd.DataFrame:
    atr_series = atr(df, window=cfg.atr_window)
    eps = 1e-12

    events = events.sort_values("idx").reset_index(drop=True)
    if cfg.use_only_valid_events and "is_valid" in events.columns:
        events = events[events["is_valid"] == True]  # noqa: E712

    rows: list[dict] = []
    for _, ev in events.iterrows():
        dir_ = ev.get("direction")
        brk_idx = int(ev["idx"])
        start = max(0, brk_idx - cfg.ob_lookback_bars)
        window = df.iloc[start:brk_idx]
        if window.empty:
            continue

        if dir_ == "up":
            cond = window["close"] < window["open"]
        else:
            cond = window["close"] > window["open"]
        body = (window["close"] - window["open"]).abs()
        rng = (window["high"] - window["low"]).replace(0, eps)
        body_ratio = body / rng
        candidates = window[cond & (body_ratio >= cfg.ob_body_ratio_min)]
        if candidates.empty:
            continue
        anchor_idx = candidates.index[-1]
        gap = brk_idx - df.index.get_loc(anchor_idx)
        if gap > cfg.ob_max_anchor_gap_bars:
            continue
        anc_row = df.loc[anchor_idx]
        atr_a = float(atr_series.loc[anchor_idx]) if anchor_idx in atr_series.index else 0.0
        atr_b = float(atr_series.iloc[brk_idx]) if brk_idx < len(atr_series) else 0.0
        pad = atr_a * cfg.ob_zone_padding_atr_mult

        if dir_ == "up":
            kind = "demand"
            top = max(float(anc_row["open"]), float(anc_row["close"]))
            bottom = float(
                anc_row["low"] if cfg.ob_prefer_full_wick else min(anc_row["open"], anc_row["close"])
            )
            top += pad
            bottom -= pad
        else:
            kind = "supply"
            bottom = min(float(anc_row["open"]), float(anc_row["close"]))
            top = float(
                anc_row["high"] if cfg.ob_prefer_full_wick else max(anc_row["open"], anc_row["close"])
            )
            top += pad
            bottom -= pad

        bottom = _round_price(bottom, cfg.tick_size, up=False)
        top = _round_price(top, cfg.tick_size, up=True)
        width = top - bottom
        atr_ref = max(eps, atr_b)
        width_atr = width / atr_ref if atr_ref > 0 else float("nan")
        if width_atr < cfg.min_width_atr or width_atr > cfg.max_width_atr:
            continue
        width_pips = width / cfg.pip_size if cfg.pip_size else float("nan")
        price_mid = (top + bottom) / 2

        future = df.iloc[brk_idx + 1 :]
        touches = future[(future["low"] <= top) & (future["high"] >= bottom)]
        times_touched = int(len(touches))
        touched = times_touched > 0
        fresh = not touched
        last_touch_time = touches.index[-1] if touched else pd.NaT

        fvg_present = False
        if cfg.fvg_detect and gap >= 1:
            a_pos = df.index.get_loc(anchor_idx)
            for j in range(a_pos, brk_idx):
                if j + 1 >= len(df):
                    break
                hi = df["high"].iloc[j]
                lo_next = df["low"].iloc[j + 1]
                lo = df["low"].iloc[j]
                hi_next = df["high"].iloc[j + 1]
                if dir_ == "up" and lo_next > hi:
                    fvg_present = True
                    break
                if dir_ == "down" and hi_next < lo:
                    fvg_present = True
                    break

        near_sweep = False
        sweep_id = np.nan
        sweep_grade = ""
        bars_to_sweep = np.nan
        if sweeps is not None and not sweeps.empty and "pen_idx" in sweeps.columns:
            sweeps = sweeps.copy()
            sweeps["bars_diff"] = sweeps["pen_idx"] - brk_idx
            mask = sweeps["bars_diff"].abs() <= 10
            if mask.any():
                sw = sweeps.loc[mask].iloc[0]
                near_sweep = True
                sweep_id = sw.get("sweep_id")
                sweep_grade = sw.get("quality_grade", "")
                bars_to_sweep = int(sw["bars_diff"])

        near_eq = False
        eq_cluster_id = np.nan
        eq_score = np.nan
        eq_side = ""
        eq_dist_atr = np.nan
        if eq_clusters is not None and not eq_clusters.empty:
            diff = (eq_clusters["price_center"] - price_mid).abs()
            idxmin = diff.idxmin()
            row_eq = eq_clusters.loc[idxmin]
            eq_cluster_id = row_eq.get("cluster_id", np.nan)
            eq_score = row_eq.get("score", np.nan)
            eq_side = row_eq.get("side", "")
            dist = float(diff.loc[idxmin])
            eq_dist_atr = dist / atr_ref if atr_ref > 0 else float("nan")
            near_eq = True

        asia_label = "none"
        if asia_daily is not None and not asia_daily.empty:
            brk_time = df.index[brk_idx]
            day = brk_time.floor("D")
            row = asia_daily[asia_daily["date"] == day]
            if not row.empty:
                asia_low = float(row.iloc[0]["asia_low"])
                asia_high = float(row.iloc[0]["asia_high"])
                if price_mid < asia_low:
                    asia_label = "below"
                elif price_mid > asia_high:
                    asia_label = "above"
                else:
                    edge_tol = atr_ref * 0.05
                    if abs(price_mid - asia_low) <= edge_tol or abs(price_mid - asia_high) <= edge_tol:
                        asia_label = "on_edge"
                    else:
                        asia_label = "inside"

        trend_state = "none"
        if (
            trend_timeline is not None
            and not trend_timeline.empty
            and "time" in trend_timeline.columns
        ):
            brk_time = df.index[brk_idx]
            mask = trend_timeline["time"] <= brk_time
            if mask.any():
                trend_state = str(trend_timeline.loc[mask].iloc[-1]["state"])

        w = cfg.score_weights or _default_weights()
        grades = cfg.grade_map or _default_grades()

        margin_norm = float(ev.get("break_margin_norm", 0.0))
        leg = float(ev.get("leg_from_origin", 0.0))
        atr_brk = float(ev.get("atr_at_break", atr_b))
        leg_norm = min(1.0, leg / atr_brk) if atr_brk > 0 else 0.0
        structure = min(1.0, 0.5 * margin_norm + 0.5 * leg_norm)

        sweep_score = 0.0
        if near_sweep:
            if str(sweep_grade) in {"A", "B"}:
                sweep_score = 1.0
            else:
                sweep_score = 0.5

        liq_eq = max(0.0, 1 - min(1.0, eq_dist_atr)) if near_eq else 0.0
        liq_asia = 1.0 if asia_label in {"inside", "on_edge"} else (0.3 if asia_label in {"above", "below"} else 0.0)
        liquidity = max(liq_eq, liq_asia)

        if kind == "demand":
            trend_score = 1.0 if trend_state == "up" else (0.5 if trend_state == "range" else 0.0)
        else:
            trend_score = 1.0 if trend_state == "down" else (0.5 if trend_state == "range" else 0.0)

        tech = 0.0
        tech += 1.0 if fvg_present else 0.0
        if fresh:
            tech += 1.0 if not touched else 0.5
        tech = min(1.0, tech / 2)

        score_total = (
            w.get("structure", 0.0) * structure
            + w.get("sweep", 0.0) * sweep_score
            + w.get("liquidity", 0.0) * liquidity
            + w.get("trend", 0.0) * trend_score
            + w.get("tech", 0.0) * tech
        )
        score_total = float(np.clip(score_total, 0.0, 1.0))
        grade = _grade(score_total, grades)

        rows.append(
            {
                "zone_id": len(rows),
                "kind": kind,
                "src_event_id": ev.get("event_id"),
                "src_event": ev.get("event"),
                "direction": dir_,
                "anchor_idx": int(df.index.get_loc(anchor_idx)),
                "anchor_time": pd.to_datetime(anchor_idx),
                "break_idx": brk_idx,
                "break_time": df.index[brk_idx],
                "price_top": top,
                "price_bottom": bottom,
                "price_mid": price_mid,
                "width": width,
                "width_pips": width_pips,
                "width_atr": width_atr,
                "atr_at_anchor": atr_a,
                "atr_at_break": atr_b,
                "fvg_present": bool(fvg_present),
                "fresh": bool(fresh),
                "touched": bool(touched),
                "times_touched": times_touched,
                "last_touch_time": last_touch_time,
                "near_sweep": bool(near_sweep),
                "sweep_id": sweep_id,
                "sweep_grade": sweep_grade,
                "bars_to_sweep": bars_to_sweep,
                "near_eq": bool(near_eq),
                "eq_cluster_id": eq_cluster_id,
                "eq_score": eq_score,
                "eq_side": eq_side,
                "eq_dist_atr": eq_dist_atr,
                "asia_label": asia_label,
                "trend_state_at_break": trend_state,
                "score_total": score_total,
                "score_structure": structure,
                "score_sweep": sweep_score,
                "score_liquidity": liquidity,
                "score_trend": trend_score,
                "score_tech": tech,
                "grade": grade,
                "notes": "",
            }
        )

    cols = [
        "zone_id",
        "kind",
        "src_event_id",
        "src_event",
        "direction",
        "anchor_idx",
        "anchor_time",
        "break_idx",
        "break_time",
        "price_top",
        "price_bottom",
        "price_mid",
        "width",
        "width_pips",
        "width_atr",
        "atr_at_anchor",
        "atr_at_break",
        "fvg_present",
        "fresh",
        "touched",
        "times_touched",
        "last_touch_time",
        "near_sweep",
        "sweep_id",
        "sweep_grade",
        "bars_to_sweep",
        "near_eq",
        "eq_cluster_id",
        "eq_score",
        "eq_side",
        "eq_dist_atr",
        "asia_label",
        "trend_state_at_break",
        "score_total",
        "score_structure",
        "score_sweep",
        "score_liquidity",
        "score_trend",
        "score_tech",
        "grade",
        "notes",
    ]
    zones = pd.DataFrame(rows, columns=cols)
    if zones.empty:
        return zones

    merged: list[dict] = []
    for row in zones.sort_values("break_idx").to_dict("records"):
        if merged:
            last = merged[-1]
            overlap = not (
                row["price_bottom"] > last["price_top"]
                or row["price_top"] < last["price_bottom"]
            )
            mid_dist = abs(row["price_mid"] - last["price_mid"])
            atr_ref = max(row["atr_at_break"], last["atr_at_break"], eps)
            if cfg.merge_overlap and (overlap or mid_dist <= atr_ref * cfg.merge_gap_atr_mult):
                last["price_bottom"] = min(last["price_bottom"], row["price_bottom"])
                last["price_top"] = max(last["price_top"], row["price_top"])
                last["price_mid"] = (last["price_bottom"] + last["price_top"]) / 2
                last["width"] = last["price_top"] - last["price_bottom"]
                last["width_pips"] = (
                    last["width"] / cfg.pip_size if cfg.pip_size else last["width_pips"]
                )
                last["width_atr"] = last["width"] / atr_ref if atr_ref > 0 else last["width_atr"]
                continue
        merged.append(row)

    merged_df = pd.DataFrame(merged, columns=cols)
    merged_df["zone_id"] = range(len(merged_df))
    return merged_df


def build_poi_segments(zones: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    end_time = df.index[-1] if not df.empty else pd.Timestamp.utcnow()
    for row in zones.itertuples():
        if row.kind == "demand":
            color = "green"
        elif row.kind == "supply":
            color = "red"
        elif row.kind == "breaker":
            color = "blue"
        else:
            color = "orange"
        records.append(
            {
                "zone_id": row.zone_id,
                "kind": row.kind,
                "y_top": row.price_top,
                "y_bottom": row.price_bottom,
                "start_time": row.anchor_time,
                "end_time": row.break_time if not pd.isna(row.break_time) else end_time,
                "color_hint": color,
            }
        )
    return pd.DataFrame(records)


def summarize_poi(zones: pd.DataFrame, cfg: PoiCfg) -> dict:
    n_total = int(len(zones))
    summary = {
        "n_zones_total": n_total,
        "n_demand": int((zones["kind"] == "demand").sum()) if n_total else 0,
        "n_supply": int((zones["kind"] == "supply").sum()) if n_total else 0,
        "n_breaker": int((zones["kind"] == "breaker").sum()) if n_total else 0,
        "n_flip": int((zones["kind"] == "flip").sum()) if n_total else 0,
        "median_width_pips": float(zones["width_pips"].median(skipna=True)) if n_total else 0.0,
        "grade_counts": zones["grade"].value_counts().to_dict() if n_total else {},
        "median_score": float(zones["score_total"].median(skipna=True)) if n_total else 0.0,
        "params": asdict(cfg),
    }
    return summary

