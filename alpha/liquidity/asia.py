from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from alpha.core.indicators import atr

Direction = Literal["up", "down", "none"]


@dataclass
class AsiaCfg:
    start_h: int = 0
    end_h: int = 8
    breakout_lookahead_h: int = 6
    confirm_with_body: bool = True
    atr_window: int = 14
    pip_size: float = 0.0001
    tick_size: float = 0.0
    tz_source: str = "UTC"


def _round_width(width: float, cfg: AsiaCfg) -> float:
    if cfg.tick_size > 0:
        return round(width / cfg.tick_size) * cfg.tick_size
    return width


def compute_asia_range_for_day(df_day: pd.DataFrame, cfg: AsiaCfg) -> dict:
    """Compute asia session range and post-session interactions for a day."""

    if "day_start" in df_day.attrs:
        day_start: pd.Timestamp = df_day.attrs["day_start"]
    else:
        day_start = df_day.index[0].normalize()

    start_ts = day_start + pd.Timedelta(hours=cfg.start_h)
    end_ts = day_start + pd.Timedelta(hours=cfg.end_h)
    day_end = day_start + pd.Timedelta(days=1)
    post_end = end_ts + pd.Timedelta(hours=cfg.breakout_lookahead_h)

    day_slice = df_day.loc[(df_day.index >= day_start) & (df_day.index < day_end)]
    asia_df = day_slice.loc[(day_slice.index >= start_ts) & (day_slice.index < end_ts)]

    record = {
        "date": day_start.date(),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "asia_high": np.nan,
        "asia_low": np.nan,
        "asia_mid": np.nan,
        "width": np.nan,
        "width_pips": np.nan,
        "width_atr_norm": np.nan,
        "time_high": pd.NaT,
        "time_low": pd.NaT,
        "post_first_touch_dir": "none",
        "post_first_break_dir": "none",
        "post_first_touch_ts": pd.NaT,
        "post_first_break_ts": pd.NaT,
        "label": "NoData",
        "notes": "",
    }

    if asia_df.empty:
        return record

    asia_high = float(asia_df["high"].max())
    asia_low = float(asia_df["low"].min())
    time_high = asia_df["high"].idxmax()
    time_low = asia_df["low"].idxmin()
    width = _round_width(asia_high - asia_low, cfg)
    asia_mid = asia_low + width / 2

    atr_series = atr(day_slice, window=cfg.atr_window)
    atr_ref = float(atr_series.iloc[-1]) if len(atr_series.dropna()) else 0.0
    eps = 1e-12
    width_atr_norm = width / max(atr_ref, eps)
    width_pips = width / cfg.pip_size if cfg.pip_size else np.nan

    record.update(
        {
            "asia_high": asia_high,
            "asia_low": asia_low,
            "asia_mid": asia_mid,
            "width": width,
            "width_pips": width_pips,
            "width_atr_norm": width_atr_norm,
            "time_high": time_high,
            "time_low": time_low,
            "label": "NoInteraction",
        }
    )

    up_edge = asia_high
    down_edge = asia_low
    post_df = df_day.loc[(df_day.index >= end_ts) & (df_day.index <= post_end)]

    first_touch_dir: Direction = "none"
    first_break_dir: Direction = "none"
    first_touch_ts = pd.NaT
    first_break_ts = pd.NaT

    for ts, bar in post_df.iterrows():
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        if cfg.confirm_with_body:
            touch_up = close >= up_edge
            touch_dn = close <= down_edge
        else:
            touch_up = high >= up_edge
            touch_dn = low <= down_edge
        break_up = close > up_edge
        break_dn = close < down_edge

        if first_touch_dir == "none" and (touch_up or touch_dn):
            first_touch_dir = "up" if touch_up else "down"
            first_touch_ts = ts
        if first_break_dir == "none" and (break_up or break_dn):
            first_break_dir = "up" if break_up else "down"
            first_break_ts = ts
        if first_touch_dir != "none" and first_break_dir != "none":
            break

    record.update(
        {
            "post_first_touch_dir": first_touch_dir,
            "post_first_break_dir": first_break_dir,
            "post_first_touch_ts": first_touch_ts,
            "post_first_break_ts": first_break_ts,
        }
    )

    if first_break_dir == "up":
        record["label"] = "BreakUp"
    elif first_break_dir == "down":
        record["label"] = "BreakDown"
    elif first_touch_dir != "none":
        record["label"] = "BothTouch_NoBreak"
    else:
        record["label"] = "NoInteraction"

    return record


def asia_range_daily(df: pd.DataFrame, cfg: AsiaCfg) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.tz_localize(cfg.tz_source)
    else:
        tz_name = str(df.index.tz)
        if cfg.tz_source and tz_name != cfg.tz_source:
            warnings.warn(
                f"tz_source {cfg.tz_source} differs from dataframe tz {tz_name}; using dataframe tz",
                RuntimeWarning,
            )

    records = []
    for day_start, _ in df.groupby(df.index.normalize()):
        post_end = day_start + pd.Timedelta(
            hours=cfg.end_h + cfg.breakout_lookahead_h
        )
        df_slice = df.loc[day_start:post_end].copy()
        df_slice.attrs["day_start"] = day_start
        record = compute_asia_range_for_day(df_slice, cfg)
        records.append(record)
    return pd.DataFrame(records)


def summarize_asia_ranges(daily_df: pd.DataFrame, cfg: AsiaCfg) -> dict:
    valid = daily_df[daily_df["label"] != "NoData"]
    n_days = int(len(daily_df))
    width_pips_stats = {
        "p25": float(valid["width_pips"].quantile(0.25)) if not valid.empty else float("nan"),
        "median": float(valid["width_pips"].median()) if not valid.empty else float("nan"),
        "p75": float(valid["width_pips"].quantile(0.75)) if not valid.empty else float("nan"),
    }
    width_atr_norm_stats = {
        "p25": float(valid["width_atr_norm"].quantile(0.25)) if not valid.empty else float("nan"),
        "median": float(valid["width_atr_norm"].median()) if not valid.empty else float("nan"),
        "p75": float(valid["width_atr_norm"].quantile(0.75)) if not valid.empty else float("nan"),
    }
    denom = len(valid) if not valid.empty else 1
    summary = {
        "n_days": n_days,
        "width_pips_stats": width_pips_stats,
        "width_atr_norm_stats": width_atr_norm_stats,
        "break_up_share": float((valid["label"] == "BreakUp").sum() / denom),
        "break_down_share": float((valid["label"] == "BreakDown").sum() / denom),
        "both_touch_share": float((valid["label"] == "BothTouch_NoBreak").sum() / denom),
        "no_interaction_share": float((valid["label"] == "NoInteraction").sum() / denom),
        "params": asdict(cfg),
    }
    return summary
