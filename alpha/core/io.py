from __future__ import annotations

"""Input/output helpers for OHLCFrame."""

from pathlib import Path
from typing import Optional

import pandas as pd

from .data import OHLCFrame, Timeframe


def read_mt_tsv(path: str, *, symbol: str, timeframe: Timeframe, tz: str = "UTC") -> OHLCFrame:
    """Read MetaTrader TSV exported data into an :class:`OHLCFrame`.

    Parameters
    ----------
    path: str
        Path to the TSV file. Expected columns ``<DATE>``, ``<TIME>``, ``<OPEN>``,
        ``<HIGH>``, ``<LOW>``, ``<CLOSE>`` among others.
    symbol: str
        Symbol name for the frame.
    timeframe: Timeframe
        Timeframe string (e.g. ``"H1"``).
    tz: str, default "UTC"
        Target timezone for the data.
    """

    df = pd.read_csv(path, sep="\t")
    # normalise column names
    df.columns = [c.strip("<>").lower() for c in df.columns]

    if "date" not in df.columns or "time" not in df.columns:
        raise ValueError("TSV missing <DATE>/<TIME> columns or wrong separator")

    # combine date and time
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
    )
    # drop invalid rows
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    cols = ["open", "high", "low", "close"]
    df = df[cols].astype("float64")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)

    ohlc = OHLCFrame(symbol=symbol, timeframe=timeframe, tz=tz, df=df)
    return ohlc


def to_parquet(ohlc: OHLCFrame, out_path: str) -> None:
    """Write OHLCFrame to a parquet file."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ohlc.df.to_parquet(out)


def from_parquet(path: str, *, symbol: str, timeframe: Timeframe, tz: str) -> OHLCFrame:
    """Load OHLCFrame from parquet file."""

    df = pd.read_parquet(path)
    ohlc = OHLCFrame(symbol=symbol, timeframe=timeframe, tz=tz, df=df)
    return ohlc
