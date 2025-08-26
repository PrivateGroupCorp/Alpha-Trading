from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from alpha.core.indicators import is_doji_series

LevelType = Literal["peak", "trough"]


@dataclass
class LevelsCfgFormation:
    doji_body_ratio: float = 0.20
    ignore_doji: bool = True
    cooldown_bars: int = 0
    min_leg_body_ratio: float = 0.0
    tick_size: float = 0.0


def _cooldown_ok(current_idx: int, last_end_idx: Optional[int], cooldown: int) -> bool:
    if cooldown <= 0:
        return True
    if last_end_idx is None:
        return True
    return current_idx - last_end_idx >= cooldown


def detect_levels_formation(df: pd.DataFrame, cfg: LevelsCfgFormation) -> pd.DataFrame:
    """Detect peak/trough levels via two-close formation."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):  # pragma: no cover - sanity check
        raise ValueError("df must contain open, high, low, close columns")

    doji = (
        is_doji_series(df, body_ratio=cfg.doji_body_ratio)
        if cfg.ignore_doji
        else pd.Series(False, index=df.index)
    )

    levels: list[dict[str, object]] = []
    last_end_idx: Optional[int] = None
    eps = 1e-12

    for i in range(1, len(df)):
        o1, h1, l1, c1 = df.iloc[i - 1][["open", "high", "low", "close"]]
        o2, h2, l2, c2 = df.iloc[i][["open", "high", "low", "close"]]

        if cfg.ignore_doji and (doji.iat[i - 1] or doji.iat[i]):
            continue

        if cfg.min_leg_body_ratio > 0:
            r1 = max(h1 - l1, eps)
            r2 = max(h2 - l2, eps)
            if abs(c1 - o1) < cfg.min_leg_body_ratio * r1:
                continue
            if abs(c2 - o2) < cfg.min_leg_body_ratio * r2:
                continue

        dir1 = 1 if c1 > o1 else (-1 if c1 < o1 else 0)
        dir2 = 1 if c2 > o2 else (-1 if c2 < o2 else 0)

        if dir1 == -1 and dir2 == -1 and c2 < c1:
            if not _cooldown_ok(i, last_end_idx, cfg.cooldown_bars):
                continue
            price = max(h1, h2)
            if cfg.tick_size > 0:
                price = round(price / cfg.tick_size) * cfg.tick_size
            levels.append(
                {
                    "time": df.index[i],
                    "type": "peak",
                    "price": float(price),
                    "start_idx": i - 1,
                    "end_idx": i,
                    "source_closes": "bear,bear",
                    "notes": "",
                }
            )
            last_end_idx = i
        elif dir1 == 1 and dir2 == 1 and c2 > c1:
            if not _cooldown_ok(i, last_end_idx, cfg.cooldown_bars):
                continue
            price = min(l1, l2)
            if cfg.tick_size > 0:
                price = round(price / cfg.tick_size) * cfg.tick_size
            levels.append(
                {
                    "time": df.index[i],
                    "type": "trough",
                    "price": float(price),
                    "start_idx": i - 1,
                    "end_idx": i,
                    "source_closes": "bull,bull",
                    "notes": "",
                }
            )
            last_end_idx = i

    return pd.DataFrame(
        levels,
        columns=[
            "time",
            "type",
            "price",
            "start_idx",
            "end_idx",
            "source_closes",
            "notes",
        ],
    )
