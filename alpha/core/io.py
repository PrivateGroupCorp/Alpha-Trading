from __future__ import annotations

"""Input/output helpers for OHLCFrame."""

from pathlib import Path
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

    required = {"date", "time", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("TSV missing required columns or wrong separator")

    # combine date and time and drop invalid rows
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    cols = ["open", "high", "low", "close"]
    df = df[cols].astype("float64")

    # handle timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    return OHLCFrame(symbol=symbol, timeframe=timeframe, tz=tz, df=df)


def to_parquet(ohlc: OHLCFrame, out_path: str) -> None:
    """Write ``OHLCFrame`` to a parquet file."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ohlc.df.to_parquet(out)


def from_parquet(path: str, *, symbol: str, timeframe: Timeframe, tz: str) -> OHLCFrame:
    """Load ``OHLCFrame`` from a parquet file."""

    df = pd.read_parquet(path)
    df = df.sort_index()

    cols = ["open", "high", "low", "close"]
    df = df[cols].astype("float64")

    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    return OHLCFrame(symbol=symbol, timeframe=timeframe, tz=tz, df=df)
