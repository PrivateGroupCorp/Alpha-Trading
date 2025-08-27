from pathlib import Path
import pandas as pd


def merge_with_existing(new_df: pd.DataFrame, out_parquet_path: Path) -> pd.DataFrame:
    if out_parquet_path.exists():
        existing = pd.read_parquet(out_parquet_path)
        df = pd.concat([existing, new_df])
    else:
        df = new_df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
