from pathlib import Path
import pandas as pd

from .base import BaseProvider, FetchRequest


class CSVLocalProvider(BaseProvider):
    """Simple provider loading OHLCV data from a local CSV/TSV file."""

    name = "csv_local"

    def fetch(self, req: FetchRequest, **kwargs) -> pd.DataFrame:
        path = Path(req.vendor_symbol)
        if not path.exists():
            raise FileNotFoundError(f"CSV path not found: {path}")

        sep = kwargs.get("sep")
        if sep is None:
            sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","

        df = pd.read_csv(path, sep=sep)
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df.set_index(time_col, inplace=True)

        # ensure expected columns
        cols = {c.lower(): c for c in df.columns}
        rename_map = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in cols:
                rename_map[cols[col]] = col
        df = df.rename(columns=rename_map)

        # filter by time range
        start = pd.Timestamp(req.start, tz="UTC")
        end = pd.Timestamp(req.end, tz="UTC")
        df = df.loc[(df.index >= start) & (df.index <= end)]

        # ensure column order
        cols_out = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols_out]
