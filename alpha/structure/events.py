from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from alpha.core.indicators import atr

Direction = Literal["up", "down"]


@dataclass
class EventsCfg:
    bos_break_mult_atr: float = 0.15
    bos_leg_min_atr_mult: float = 0.55
    atr_window: int = 14
    event_cooldown_bars: int = 3
    confirm_with_body_only: bool = True


def detect_bos_choch(
    df: pd.DataFrame,
    swings_df: pd.DataFrame,
    cfg: EventsCfg,
) -> pd.DataFrame:
    """Detect BOS/CHoCH events from OHLC data and swings."""

    atr_series = atr(df, window=cfg.atr_window)
    swings = swings_df.sort_values("idx").reset_index(drop=True)
    eps = 1e-12

    last_peak: dict | None = None
    last_trough: dict | None = None
    sw_ptr = 0
    events: list[dict] = []
    last_bos_dir: Direction | str = "none"
    last_event_idx = -10**9
    start_idx = int(swings["idx"].min()) if not swings.empty else 0

    for j in range(start_idx, len(df)):
        while sw_ptr < len(swings) and int(swings.loc[sw_ptr, "idx"]) <= j:
            row = swings.loc[sw_ptr]
            rec = {
                "swing_id": int(row["swing_id"]),
                "idx": int(row["idx"]),
                "price": float(row["price"]),
            }
            if row["type"] == "peak":
                last_peak = rec
            else:
                last_trough = rec
            sw_ptr += 1

        if last_peak is None or last_trough is None:
            continue
        if j - last_event_idx < cfg.event_cooldown_bars:
            continue

        atr_j = float(atr_series.iloc[j]) if j < len(atr_series) else 0.0
        thr = max(eps, atr_j * cfg.bos_break_mult_atr)
        close_j = float(df["close"].iloc[j])
        price_up = close_j if cfg.confirm_with_body_only else float(df["high"].iloc[j])
        price_dn = close_j if cfg.confirm_with_body_only else float(df["low"].iloc[j])

        cond_up = (
            price_up > last_peak["price"] + thr
            and (close_j - last_trough["price"]) >= atr_j * cfg.bos_leg_min_atr_mult
        )
        cond_dn = (
            price_dn < last_trough["price"] - thr
            and (last_peak["price"] - close_j) >= atr_j * cfg.bos_leg_min_atr_mult
        )
        if not (cond_up or cond_dn):
            continue

        direction: Direction = "up" if cond_up else "down"
        ref = last_peak if direction == "up" else last_trough
        origin = last_trough if direction == "up" else last_peak
        margin = (
            close_j - (ref["price"] + thr)
            if direction == "up"
            else (ref["price"] - thr) - close_j
        )
        prev_dir = last_bos_dir
        is_choch = last_bos_dir != "none" and direction != last_bos_dir
        event_type = "CHoCH" if is_choch else "BOS"

        events.append(
            {
                "event_id": len(events),
                "time": df.index[j],
                "idx": j,
                "event": event_type,
                "direction": direction,
                "ref_swing_id": ref["swing_id"],
                "ref_price": ref["price"],
                "origin_swing_id": origin["swing_id"],
                "origin_price": origin["price"],
                "leg_from_origin": abs(close_j - origin["price"]),
                "atr_at_break": atr_j,
                "confirm_threshold": thr,
                "break_margin": margin,
                "break_margin_norm": max(0.0, margin) / max(eps, thr),
                "bars_since_ref": j - ref["idx"],
                "bars_since_origin": j - origin["idx"],
                "prev_bos_direction": prev_dir,
            }
        )
        last_bos_dir = direction
        last_event_idx = j

    cols = [
        "event_id",
        "time",
        "idx",
        "event",
        "direction",
        "ref_swing_id",
        "ref_price",
        "origin_swing_id",
        "origin_price",
        "leg_from_origin",
        "atr_at_break",
        "confirm_threshold",
        "break_margin",
        "break_margin_norm",
        "bars_since_ref",
        "bars_since_origin",
        "prev_bos_direction",
    ]
    events_df = pd.DataFrame(events, columns=cols)
    if not events_df.empty:
        events_df["time"] = pd.to_datetime(events_df["time"], utc=True)
    return events_df


def summarize_events(events_df: pd.DataFrame, cfg: EventsCfg) -> dict:
    """Summarize events for reporting."""

    n_events = int(len(events_df))
    n_bos = int((events_df["event"] == "BOS").sum()) if n_events else 0
    n_choch = int((events_df["event"] == "CHoCH").sum()) if n_events else 0
    up_count = int((events_df["direction"] == "up").sum()) if n_events else 0
    down_count = int((events_df["direction"] == "down").sum()) if n_events else 0
    if n_events > 1:
        bars_between = events_df["idx"].diff().iloc[1:]
        median_between = float(bars_between.median(skipna=True))
    else:
        median_between = 0.0
    median_margin = (
        float(events_df["break_margin_norm"].median(skipna=True)) if n_events else 0.0
    )
    return {
        "n_events": n_events,
        "n_bos": n_bos,
        "n_choch": n_choch,
        "up": up_count,
        "down": down_count,
        "median_bars_between_events": median_between,
        "median_break_margin_norm": median_margin,
        "params": {
            "bos_break_mult_atr": cfg.bos_break_mult_atr,
            "bos_leg_min_atr_mult": cfg.bos_leg_min_atr_mult,
            "atr_window": cfg.atr_window,
            "event_cooldown_bars": cfg.event_cooldown_bars,
            "confirm_with_body_only": cfg.confirm_with_body_only,
        },
    }
