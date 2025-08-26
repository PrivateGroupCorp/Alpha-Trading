from __future__ import annotations

"""Data contract definitions for alpha trading package."""

from pydantic import BaseModel
from typing import Literal
import pandas as pd

Timeframe = Literal["H1", "M15", "M1"]


class OHLCFrame(BaseModel):
    """Container for OHLC time series data."""

    symbol: str
    timeframe: Timeframe
    tz: str
    df: pd.DataFrame  # DatetimeIndex, columns: open, high, low, close

    class Config:
        arbitrary_types_allowed = True
