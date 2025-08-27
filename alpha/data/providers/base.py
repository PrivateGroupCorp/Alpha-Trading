from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class FetchRequest:
    symbol: str
    vendor_symbol: str
    tf: str
    start: str
    end: str
    tz: str
    chunk_days: int = 7


class BaseProvider:
    """Base interface for data providers."""

    name: str = "base"

    def fetch(self, req: FetchRequest, **kwargs) -> pd.DataFrame:
        """Fetch OHLCV data for the given request.

        The returned dataframe must have columns open, high, low, close
        and optionally volume. Index must be datetime-like.
        """
        raise NotImplementedError
