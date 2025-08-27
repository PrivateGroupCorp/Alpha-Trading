from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict

import pandas as pd

TriggerName = Literal["touch_reject", "choch_bos_in_zone"]


@dataclass
class TriggerCfgTouch:
    max_pen_atr_mult: float = 0.20
    min_reject_wick_ratio: float = 0.4
    confirm_with_body_against: bool = True


@dataclass
class TriggerCfgChoch:
    min_internal_leg_atr_mult: float = 0.10
    bos_confirm_k: float = 0.10
    lookahead_bars: int = 6


def _wick_ratio(bar: pd.Series, direction: str) -> float:
    """Return wick ratio for candle in direction of rejection."""

    rng = max(bar["high"] - bar["low"], 1e-12)
    if direction == "long":
        wick = min(bar["open"], bar["close"]) - bar["low"]
    else:
        wick = bar["high"] - max(bar["open"], bar["close"])
    return wick / rng


def check_touch_reject(
    bar: pd.Series,
    zone_row: pd.Series,
    atr_value: float,
    cfg: TriggerCfgTouch,
) -> bool:
    """Simple touch/reject trigger inside a POI/OB zone."""

    top = float(zone_row["y_top"])
    bottom = float(zone_row["y_bottom"])
    side = str(zone_row.get("side", "long"))

    penetrated = False
    if side == "long":
        if bar["low"] <= top and bar["high"] >= bottom:
            penetration = max(0.0, top - bar["low"])
            penetrated = penetration <= atr_value * cfg.max_pen_atr_mult
    else:
        if bar["high"] >= bottom and bar["low"] <= top:
            penetration = max(0.0, bar["high"] - bottom)
            penetrated = penetration <= atr_value * cfg.max_pen_atr_mult

    if not penetrated:
        return False

    wick_ratio = _wick_ratio(bar, side)
    if wick_ratio < cfg.min_reject_wick_ratio:
        return False

    if cfg.confirm_with_body_against:
        if side == "long" and bar["close"] < top:
            return False
        if side == "short" and bar["close"] > bottom:
            return False

    return True


def check_choch_in_zone(
    m1_df: pd.DataFrame,
    j: int,
    zone_row: pd.Series,
    atr_series: pd.Series,
    cfg: TriggerCfgChoch,
) -> bool:
    """Simplified CHoCH/BOS detection inside zone."""

    side = str(zone_row.get("side", "long"))
    top = float(zone_row["y_top"])
    bottom = float(zone_row["y_bottom"])
    atr_val = float(atr_series.iloc[j])

    lookahead = min(cfg.lookahead_bars, len(m1_df) - j - 1)
    if lookahead <= 2:
        return False

    highs = m1_df["high"].iloc[j : j + lookahead + 1]
    lows = m1_df["low"].iloc[j : j + lookahead + 1]
    closes = m1_df["close"].iloc[j : j + lookahead + 1]

    if side == "long":
        prev_high = highs.iloc[0]
        min_low = lows.min()
        if prev_high - min_low < atr_val * cfg.min_internal_leg_atr_mult:
            return False
        bos_level = prev_high + atr_val * cfg.bos_confirm_k
        return bool((closes[1:] >= bos_level).any())
    else:
        prev_low = lows.iloc[0]
        max_high = highs.max()
        if max_high - prev_low < atr_val * cfg.min_internal_leg_atr_mult:
            return False
        bos_level = prev_low - atr_val * cfg.bos_confirm_k
        return bool((closes[1:] <= bos_level).any())
