from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from alpha.core.indicators import atr

PropRefMode = Literal["prev_leg", "atr", "prev_leg_or_atr"]


@dataclass
class LevelsCfgProportionality:
    proportionality_ratio: float = 0.5
    prop_ref_mode: PropRefMode = "prev_leg_or_atr"
    atr_window: int = 14


def compute_levels_proportionality(
    df: pd.DataFrame,
    levels_formation: pd.DataFrame,
    cfg: LevelsCfgProportionality,
) -> pd.DataFrame:
    """Append proportionality metrics to ``levels_formation``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC dataframe with ``open``, ``high``, ``low`` and ``close`` columns.
    levels_formation : pd.DataFrame
        Two-close formation levels.
    cfg : LevelsCfgProportionality
        Configuration with ratio, reference mode and ATR window.

    Returns
    -------
    pd.DataFrame
        ``levels_formation`` with extra columns:
        ``leg_current``, ``leg_ref``, ``weak_prop``, ``prop_ref_mode``,
        ``proportionality_ratio``.
    """

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):  # pragma: no cover - sanity check
        raise ValueError("df must contain open, high, low, close columns")

    out_cols = list(levels_formation.columns) + [
        "leg_current",
        "leg_ref",
        "weak_prop",
        "prop_ref_mode",
        "proportionality_ratio",
    ]

    if levels_formation.empty:
        return pd.DataFrame(columns=out_cols)

    atr_series = atr(df, window=cfg.atr_window)
    rows: list[dict[str, object]] = []
    prev_leg_val = np.nan

    for _, lvl in levels_formation.sort_values("time").iterrows():
        si = int(lvl["start_idx"])
        ei = int(lvl["end_idx"])
        leg_current = float(abs(df["close"].iloc[ei] - df["close"].iloc[si]))
        atr_val = float(atr_series.iloc[ei])

        if cfg.prop_ref_mode == "prev_leg":
            leg_ref = prev_leg_val if not np.isnan(prev_leg_val) else atr_val
        elif cfg.prop_ref_mode == "atr":
            leg_ref = atr_val
        else:  # prev_leg_or_atr
            base = prev_leg_val if not np.isnan(prev_leg_val) else 0.0
            leg_ref = max(base, atr_val)

        weak_prop = bool(leg_current < cfg.proportionality_ratio * leg_ref)

        rows.append(
            {
                **lvl,
                "leg_current": leg_current,
                "leg_ref": leg_ref,
                "weak_prop": weak_prop,
                "prop_ref_mode": cfg.prop_ref_mode,
                "proportionality_ratio": cfg.proportionality_ratio,
            }
        )
        prev_leg_val = leg_current

    return pd.DataFrame(rows, columns=out_cols)
