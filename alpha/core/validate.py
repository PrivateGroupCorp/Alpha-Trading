from __future__ import annotations

"""Validation helpers for :class:`~alpha.core.data.OHLCFrame`."""

from typing import Dict

import pandas as pd

from .data import OHLCFrame

# mapping timeframe to expected minutes
_TIMEFRAME_MINUTES = {"H1": 60, "M15": 15, "M1": 1}


def validate_ohlc_frame(ohlc: OHLCFrame) -> Dict[str, object]:
    """Validate OHLCFrame and return summary statistics.

    Parameters
    ----------
    ohlc: OHLCFrame
        Frame to validate. ``ohlc.df`` may be modified (duplicate removal).

    Returns
    -------
    dict
        Summary information used for ``meta.json``.
    """

    df = ohlc.df.copy()

    cols = ["open", "high", "low", "close"]

    # ensure proper dtype and no NaNs
    if not all(pd.api.types.is_float_dtype(df[c]) for c in cols):
        raise ValueError("OHLC columns must be float")

    null_counts = df[cols].isna().sum().to_dict()
    if any(null_counts.values()):
        raise ValueError("NaN values in OHLC columns")

    # sort index and drop duplicates
    df = df.sort_index()
    dup_count = int(df.index.duplicated().sum())
    if dup_count:
        df = df[~df.index.duplicated(keep="first")]

    # monotonic check
    if not df.index.is_monotonic_increasing:
        raise ValueError("Datetime index must be monotonic increasing")

    # gap detection
    freq_minutes = _TIMEFRAME_MINUTES.get(ohlc.timeframe)
    expected_delta = pd.Timedelta(minutes=freq_minutes)
    diffs = df.index.to_series().diff().dropna()
    gap_deltas = diffs[diffs > expected_delta]
    missing = int(((gap_deltas / expected_delta) - 1).sum())
    gaps_summary = {"gaps": int(len(gap_deltas)), "missing_rows": missing}

    ohlc.df = df

    summary = {
        "n_rows": int(len(df)),
        "start": df.index.min().isoformat() if not df.empty else None,
        "end": df.index.max().isoformat() if not df.empty else None,
        "null_counts": null_counts,
        "duplicate_rows": dup_count,
        "gaps_summary": gaps_summary,
    }

    return summary
