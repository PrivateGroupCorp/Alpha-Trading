import pandas as pd
import numpy as np
from typing import Literal

ATRMethod = Literal["ewm", "wilder"]

__all__ = [
    "ATRMethod",
    "true_range",
    "atr_ewm",
    "atr_wilder",
    "is_doji_row",
    "is_doji_series",
    "atr",
]


def _ensure_float64(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, ...]:
    """Return requested columns as float64 series."""
    return tuple(df[c].astype("float64") for c in cols)


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute the True Range of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``high``, ``low`` and ``close`` columns.

    Returns
    -------
    pd.Series
        True range values aligned with ``df.index``.
    """
    high, low, close = _ensure_float64(df, ["high", "low", "close"])
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = tr.fillna(tr1)
    tr.name = "tr"
    return tr


def atr_ewm(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using exponential weighting.

    ``alpha`` is set to ``1 / window`` and ``adjust=False``.
    """
    tr = true_range(df)
    atr = tr.ewm(alpha=1.0 / float(window), adjust=False).mean()
    atr.name = f"atr_ewm_{window}"
    return atr


def atr_wilder(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    tr = true_range(df).to_numpy(dtype="float64")
    atr = np.full_like(tr, np.nan, dtype="float64")

    if len(tr) >= window:
        atr[window - 1] = tr[:window].mean()
        for i in range(window, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

    out = pd.Series(atr, index=df.index, name=f"atr_wilder_{window}")
    return out


def is_doji_row(o: float, h: float, lo: float, c: float, body_ratio: float = 0.20) -> bool:
    """Detect a doji candlestick for a single row."""
    eps = 1e-12
    body = abs(c - o)
    rng = max(h - lo, eps)
    return body <= body_ratio * rng


def is_doji_series(df: pd.DataFrame, body_ratio: float = 0.20) -> pd.Series:
    """Vectorised doji detection for a dataframe."""
    o, h, lo, c = _ensure_float64(df, ["open", "high", "low", "close"])
    eps = 1e-12
    body = (c - o).abs()
    rng = np.maximum(h - lo, eps)
    doji = body <= body_ratio * rng
    return pd.Series(doji, index=df.index, name="is_doji", dtype="bool")


def atr(df: pd.DataFrame, window: int = 14, method: ATRMethod = "ewm") -> pd.Series:
    """Wrapper for ATR calculation selecting the smoothing method."""
    if method == "ewm":
        return atr_ewm(df, window)
    if method == "wilder":
        return atr_wilder(df, window)
    raise ValueError("Unknown ATR method")
