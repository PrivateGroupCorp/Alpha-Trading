from __future__ import annotations

"""Data contract definitions for alpha trading package."""

from pydantic import BaseModel, ConfigDict
from typing import Literal
import pandas as pd

# Supported timeframes
Timeframe = Literal["H1", "M15", "M1"]


class OHLCFrame(BaseModel):
    """Container for OHLC time series data.

    ``df`` is expected to contain a ``DatetimeIndex`` and the columns
    ``open``, ``high``, ``low`` and ``close`` all of type ``float64``.
    ``OHLCFrame`` itself does not perform validation; this is handled by
    :func:`alpha.core.validate.validate_ohlc_frame`.
    """

    symbol: str
    timeframe: Timeframe
    tz: str
    df: pd.DataFrame  # DatetimeIndex, columns: open, high, low, close

    # Allow pandas DataFrame as a field type
    model_config = ConfigDict(arbitrary_types_allowed=True)
