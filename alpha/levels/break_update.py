from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from alpha.core.indicators import atr

BreakMode = Literal["atr", "pips"]


@dataclass
class LevelsCfgBreakUpdate:
    break_mode: BreakMode = "atr"
    atr_mult: float = 0.15
    pips: float = 10.0
    update_on_wick: bool = True
    max_update_distance_atr_mult: float = 2.0
    tick_size: float = 0.0
    pip_size: float = 0.0001
    atr_window: int = 14


def apply_break_update(
    df: pd.DataFrame,
    levels_df: pd.DataFrame,
    cfg: LevelsCfgBreakUpdate,
) -> pd.DataFrame:
    """Determine break/update state for each level.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC dataframe with a ``DatetimeIndex``.
    levels_df : pd.DataFrame
        Levels dataframe from formation/proportionality step.
    cfg : LevelsCfgBreakUpdate
        Configuration for break/update logic.

    Returns
    -------
    pd.DataFrame
        ``levels_df`` with additional state columns.
    """

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):  # pragma: no cover - sanity check
        raise ValueError("df must contain open, high, low, close columns")

    levels = levels_df.copy().reset_index(drop=True)
    n = len(df)
    atr_series = atr(df, window=cfg.atr_window)
    eps = 1e-12

    # initialise new columns
    levels["state"] = "intact"
    levels["break_time"] = pd.Series(pd.NaT, index=levels.index, dtype="datetime64[ns, UTC]")
    levels["break_idx"] = -1
    levels["last_extreme"] = levels["price"].astype(float)
    levels["update_count"] = 0
    levels["touch_count"] = 0
    levels["first_touch_time"] = pd.Series(
        pd.NaT, index=levels.index, dtype="datetime64[ns, UTC]"
    )
    levels["confirm_threshold"] = 0.0
    levels["break_mode"] = cfg.break_mode
    levels["tick_size"] = cfg.tick_size

    for idx, row in levels.iterrows():
        price = float(row["price"])
        last_extreme = price
        update_count = 0
        touch_count = 0
        first_touch_time = pd.NaT
        state = "intact"
        break_idx = -1
        break_time = pd.NaT

        end_idx = int(row["end_idx"])
        start_scan = end_idx + 1
        if start_scan < n:
            atr_j = float(atr_series.iat[start_scan])
        else:
            atr_j = float(atr_series.iat[end_idx])
        atr_j = max(atr_j, eps)
        confirm_thr = (
            atr_j * cfg.atr_mult
            if cfg.break_mode == "atr"
            else cfg.pips * cfg.pip_size
        )

        for j in range(start_scan, n):
            atr_j = max(float(atr_series.iat[j]), eps)
            thr = (
                atr_j * cfg.atr_mult
                if cfg.break_mode == "atr"
                else cfg.pips * cfg.pip_size
            )
            confirm_thr = thr

            close_j = float(df["close"].iat[j])
            high_j = float(df["high"].iat[j])
            low_j = float(df["low"].iat[j])

            if row["type"] == "peak":
                if close_j > price + thr:
                    state = "broken"
                    break_idx = j
                    break_time = df.index[j]
                    if cfg.tick_size > 0:
                        price = round(price / cfg.tick_size) * cfg.tick_size
                    break
                if cfg.update_on_wick and high_j >= price and close_j <= price + thr:
                    if touch_count == 0:
                        first_touch_time = df.index[j]
                    touch_count += 1
                    new_price = max(last_extreme, high_j)
                    if abs(new_price - price) <= atr_j * cfg.max_update_distance_atr_mult:
                        price = last_extreme = new_price
                        update_count += 1
                        if cfg.tick_size > 0:
                            price = last_extreme = round(
                                price / cfg.tick_size
                            ) * cfg.tick_size
            else:  # trough
                if close_j < price - thr:
                    state = "broken"
                    break_idx = j
                    break_time = df.index[j]
                    if cfg.tick_size > 0:
                        price = round(price / cfg.tick_size) * cfg.tick_size
                    break
                if cfg.update_on_wick and low_j <= price and close_j >= price - thr:
                    if touch_count == 0:
                        first_touch_time = df.index[j]
                    touch_count += 1
                    new_price = min(last_extreme, low_j)
                    if abs(new_price - price) <= atr_j * cfg.max_update_distance_atr_mult:
                        price = last_extreme = new_price
                        update_count += 1
                        if cfg.tick_size > 0:
                            price = last_extreme = round(
                                price / cfg.tick_size
                            ) * cfg.tick_size

        levels.at[idx, "price"] = price
        levels.at[idx, "state"] = state
        levels.at[idx, "break_idx"] = break_idx
        levels.at[idx, "break_time"] = break_time
        levels.at[idx, "last_extreme"] = last_extreme
        levels.at[idx, "update_count"] = update_count
        levels.at[idx, "touch_count"] = touch_count
        levels.at[idx, "first_touch_time"] = first_touch_time
        levels.at[idx, "confirm_threshold"] = confirm_thr
        levels.at[idx, "break_mode"] = cfg.break_mode
        levels.at[idx, "tick_size"] = cfg.tick_size

    return levels
