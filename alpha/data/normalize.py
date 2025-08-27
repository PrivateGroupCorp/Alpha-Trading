from dataclasses import dataclass
import pandas as pd


@dataclass
class NormCfg:
    tz: str = "UTC"
    drop_incomplete_ohlc: bool = True
    fill_volume_nan_with_zero: bool = True
    enforce_monotonic: bool = True
    clip_negative_prices: bool = True


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in cols:
            rename_map[cols[col]] = col
    df = df.rename(columns=rename_map)
    return df


def normalize_ohlc(df: pd.DataFrame, cfg: NormCfg) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_columns(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(cfg.tz)

    ohlc_cols = ["open", "high", "low", "close"]
    if cfg.drop_incomplete_ohlc:
        df = df.dropna(subset=[c for c in ohlc_cols if c in df.columns])

    if "volume" in df.columns and cfg.fill_volume_nan_with_zero:
        df["volume"] = df["volume"].fillna(0)

    if cfg.clip_negative_prices:
        for col in ohlc_cols:
            if col in df.columns:
                df = df[df[col] > 0]

    df = df.sort_index()
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]

    if cfg.enforce_monotonic and not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # ensure float type
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
