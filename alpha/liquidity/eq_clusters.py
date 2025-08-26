from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np
import pandas as pd

from alpha.core.indicators import atr

Side = Literal["eqh", "eql"]


@dataclass
class EqCfg:
    eq_atr_tol: float = 0.06
    alt_pip_tol: float = 0.0
    body_confirm: bool = False
    max_dupe_bars: int = 1
    use_swings_gate: bool = False

    merge_overlap: bool = True
    merge_gap_atr_mult: float = 0.25
    min_touch_count: int = 2
    min_age_bars: int = 5

    pip_size: float = 0.0001
    tick_size: float = 0.0
    atr_window: int = 14
    annotate_asia: bool = True

    weights: dict = field(
        default_factory=lambda: {
            "touches": 0.40,
            "age": 0.20,
            "tightness": 0.25,
            "cleanliness": 0.15,
        }
    )


def detect_eq_touches(
    df: pd.DataFrame,
    cfg: EqCfg,
    swings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect equal high/low touches in ``df``."""

    atr_series = atr(df, window=cfg.atr_window)
    eps = 1e-12

    # Prepare swings gating arrays
    swings_peak = np.array([])
    swings_trough = np.array([])
    if cfg.use_swings_gate and swings_df is not None:
        if "price" in swings_df.columns and "type" in swings_df.columns:
            swings_peak = swings_df[swings_df["type"] == "peak"]["price"].to_numpy()
            swings_trough = swings_df[swings_df["type"] == "trough"]["price"].to_numpy()

    refs: dict[Side, list[dict]] = {"eqh": [], "eql": []}

    for j, (ts, row) in enumerate(df.iterrows()):
        atr_j = float(atr_series.iloc[j]) if j < len(atr_series) else 0.0
        tol = max(cfg.alt_pip_tol * cfg.pip_size, atr_j * cfg.eq_atr_tol, eps)
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        for side, price in (("eqh", high), ("eql", low)):
            is_body = abs(close - price) <= tol
            if cfg.body_confirm and not is_body:
                continue

            near_swing = False
            if cfg.use_swings_gate:
                arr = swings_peak if side == "eqh" else swings_trough
                if arr.size:
                    near_swing = np.min(np.abs(arr - price)) <= tol
                if not near_swing:
                    continue

            ref_list = refs[side]
            matched = None
            for ref in ref_list:
                if abs(price - ref["price"]) <= tol:
                    if j - ref["last_idx"] <= cfg.max_dupe_bars:
                        matched = None
                        break
                    matched = ref
                    break
            if matched is None:
                matched = {"price": price, "last_idx": j, "touches": []}
                ref_list.append(matched)
            else:
                matched["last_idx"] = j

            matched["touches"].append(
                {
                    "time": ts,
                    "idx": j,
                    "side": side,
                    "ref_price": matched["price"],
                    "high": high,
                    "low": low,
                    "close": close,
                    "atr": atr_j,
                    "tol_price": tol,
                    "is_body_confirmed": is_body,
                    "near_swing": near_swing,
                }
            )

    records: list[dict] = []
    for side_refs in refs.values():
        for ref in side_refs:
            if len(ref["touches"]) >= 2:
                records.extend(ref["touches"])

    if not records:
        return pd.DataFrame(
            columns=[
                "time",
                "idx",
                "side",
                "ref_price",
                "high",
                "low",
                "close",
                "atr",
                "tol_price",
                "is_body_confirmed",
                "near_swing",
            ]
        )

    out = pd.DataFrame(records)
    out.sort_values("time", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def build_eq_clusters(
    df: pd.DataFrame,
    touches: pd.DataFrame,
    cfg: EqCfg,
    asia_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build clusters from eq touches."""

    if touches.empty:
        out = pd.DataFrame(
            columns=[
                "cluster_id",
                "side",
                "price_center",
                "price_min",
                "price_max",
                "width",
                "width_pips",
                "width_atr",
                "touch_count",
                "first_time",
                "last_time",
                "age_bars",
                "clean_bodies_ratio",
                "density_per_1000",
            ]
        )
        out.attrs["n_total"] = 0
        if cfg.annotate_asia and asia_df is not None:
            out["asia_label"] = []
        return out

    clusters_all: list[dict] = []

    for side in ["eqh", "eql"]:
        side_df = touches[touches["side"] == side].copy()
        if side_df.empty:
            continue
        side_df["low"] = side_df["ref_price"] - side_df["tol_price"]
        side_df["high"] = side_df["ref_price"] + side_df["tol_price"]
        side_df.sort_values("ref_price", inplace=True)

        cur: dict | None = None
        for row in side_df.itertuples():
            interval = (row.low, row.high)
            center = float(row.ref_price)
            atr_j = float(row.atr)
            if cur is None:
                cur = {
                    "side": side,
                    "touches": [row],
                    "price_min": interval[0],
                    "price_max": interval[1],
                }
                continue
            overlap = cur["price_min"] <= interval[1] and interval[0] <= cur["price_max"]
            center_cur = np.median([t.ref_price for t in cur["touches"]])
            gap = abs(center - center_cur)
            merge_gap = gap <= atr_j * cfg.merge_gap_atr_mult
            if cfg.merge_overlap and overlap or merge_gap:
                cur["touches"].append(row)
                cur["price_min"] = min(cur["price_min"], interval[0])
                cur["price_max"] = max(cur["price_max"], interval[1])
            else:
                clusters_all.append(cur)
                cur = {
                    "side": side,
                    "touches": [row],
                    "price_min": interval[0],
                    "price_max": interval[1],
                }
        if cur is not None:
            clusters_all.append(cur)

    total_clusters = len(clusters_all)
    records: list[dict] = []
    eps = 1e-12
    for cluster in clusters_all:
        touches_list = cluster["touches"]
        ref_prices = np.array([t.ref_price for t in touches_list], dtype=float)
        atr_vals = np.array([t.atr for t in touches_list], dtype=float)
        idxs = np.array([t.idx for t in touches_list], dtype=int)
        times = [t.time for t in touches_list]
        price_center = float(np.median(ref_prices))
        price_min = float(cluster["price_min"])
        price_max = float(cluster["price_max"])
        width = price_max - price_min
        atr_mean = float(atr_vals.mean()) if len(atr_vals) else 0.0
        width_atr = width / max(atr_mean, eps)
        width_pips = width / cfg.pip_size if cfg.pip_size else np.nan
        touch_count = int(len(touches_list))
        first_idx = int(idxs.min())
        last_idx = int(idxs.max())
        first_time = times[np.argmin(idxs)]
        last_time = times[np.argmax(idxs)]
        age_bars = last_idx - first_idx + 1
        slice_df = df.iloc[first_idx : last_idx + 1]
        if cluster["side"] == "eqh":
            clean = float((slice_df["close"] <= price_max).sum()) / len(slice_df)
        else:
            clean = float((slice_df["close"] >= price_min).sum()) / len(slice_df)
        density = touch_count / max(age_bars, 1) * 1000.0

        record = {
            "side": cluster["side"],
            "price_center": price_center,
            "price_min": price_min,
            "price_max": price_max,
            "width": width,
            "width_pips": width_pips,
            "width_atr": width_atr,
            "touch_count": touch_count,
            "first_time": first_time,
            "last_time": last_time,
            "age_bars": age_bars,
            "clean_bodies_ratio": clean,
            "density_per_1000": density,
        }

        if cfg.annotate_asia and asia_df is not None and "date" in asia_df.columns:
            day = pd.to_datetime(first_time).date()
            row = asia_df.loc[asia_df["date"] == pd.to_datetime(day)]
            label = "none"
            if not row.empty:
                asia_high = float(row.iloc[0]["asia_high"])
                asia_low = float(row.iloc[0]["asia_low"])
                edge_tol = atr_mean * cfg.eq_atr_tol
                if price_center > asia_high + edge_tol:
                    label = "above"
                elif price_center < asia_low - edge_tol:
                    label = "below"
                elif (
                    abs(price_center - asia_high) <= edge_tol
                    or abs(price_center - asia_low) <= edge_tol
                ):
                    label = "on_edge"
                else:
                    label = "inside"
            record["asia_label"] = label

        records.append(record)

    clusters_df = pd.DataFrame(records)
    clusters_df.attrs["n_total"] = total_clusters

    if clusters_df.empty:
        clusters_df["cluster_id"] = []
        return clusters_df

    valid = clusters_df[
        (clusters_df["touch_count"] >= cfg.min_touch_count)
        & (clusters_df["age_bars"] >= cfg.min_age_bars)
    ].copy()
    valid.reset_index(drop=True, inplace=True)
    valid["cluster_id"] = range(len(valid))
    return valid


def score_clusters(clusters: pd.DataFrame, cfg: EqCfg) -> pd.DataFrame:
    if clusters.empty:
        for col in [
            "score",
            "score_touches",
            "score_age",
            "score_tightness",
            "score_cleanliness",
        ]:
            clusters[col] = []
        return clusters

    eps = 1e-12
    p95_touches = clusters["touch_count"].quantile(0.95)
    p95_age = clusters["age_bars"].quantile(0.95)
    p95_width = clusters["width_atr"].quantile(0.95)

    clusters = clusters.copy()
    clusters["score_touches"] = (
        np.log1p(clusters["touch_count"]) / max(np.log1p(p95_touches), eps)
    ).clip(0, 1)
    clusters["score_age"] = (clusters["age_bars"] / max(p95_age, eps)).clip(0, 1)
    clusters["score_tightness"] = (1 - clusters["width_atr"] / max(p95_width, eps)).clip(0, 1)
    clusters["score_cleanliness"] = clusters["clean_bodies_ratio"].clip(0, 1)
    w = cfg.weights
    clusters["score"] = (
        clusters["score_touches"] * w.get("touches", 0.0)
        + clusters["score_age"] * w.get("age", 0.0)
        + clusters["score_tightness"] * w.get("tightness", 0.0)
        + clusters["score_cleanliness"] * w.get("cleanliness", 0.0)
    )
    return clusters


def summarize_eq(clusters: pd.DataFrame, touches: pd.DataFrame, cfg: EqCfg) -> dict:
    n_touches = int(len(touches))
    n_total = int(clusters.attrs.get("n_total", len(clusters)))
    n_valid = int(len(clusters))
    width_pips_stats = {
        "p25": float(clusters["width_pips"].quantile(0.25)) if not clusters.empty else float("nan"),
        "median": float(clusters["width_pips"].median()) if not clusters.empty else float("nan"),
        "p75": float(clusters["width_pips"].quantile(0.75)) if not clusters.empty else float("nan"),
    }
    width_atr_stats = {
        "p25": float(clusters["width_atr"].quantile(0.25)) if not clusters.empty else float("nan"),
        "median": float(clusters["width_atr"].median()) if not clusters.empty else float("nan"),
        "p75": float(clusters["width_atr"].quantile(0.75)) if not clusters.empty else float("nan"),
    }
    denom = len(clusters) if not clusters.empty else 1
    share_eqh = float((clusters["side"] == "eqh").sum() / denom)
    share_eql = float((clusters["side"] == "eql").sum() / denom)
    summary = {
        "n_touches": n_touches,
        "n_clusters_total": n_total,
        "n_clusters_valid": n_valid,
        "width_pips_stats": width_pips_stats,
        "width_atr_stats": width_atr_stats,
        "share_eqh": share_eqh,
        "share_eql": share_eql,
        "touch_count_mean": float(clusters["touch_count"].mean())
        if not clusters.empty
        else float("nan"),
        "touch_count_median": float(clusters["touch_count"].median())
        if not clusters.empty
        else float("nan"),
        "score_mean": float(clusters["score"].mean()) if "score" in clusters else float("nan"),
        "score_median": float(clusters["score"].median()) if "score" in clusters else float("nan"),
        "params": asdict(cfg),
    }
    return summary
