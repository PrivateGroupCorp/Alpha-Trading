"""Data quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run(path: Path) -> Dict[str, Any]:
    """Run basic data quality checks on ohlc.parquet file."""
    result: Dict[str, Any] = {"exists": False}
    ohlc_path = path / "ohlc.parquet"
    if not ohlc_path.exists():
        return result
    df = pd.read_parquet(ohlc_path)
    result["exists"] = True
    result["n_rows"] = len(df)
    result["dup_pct"] = 0.0
    if "time" in df.columns:
        result["dup_pct"] = (
            df["time"].duplicated().sum() / max(len(df), 1) * 100.0
        )
    result["nan_rows_pct"] = df.isna().any(axis=1).mean() * 100.0
    if "time" in df.columns:
        times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        times = pd.to_datetime(df.index, utc=True, errors="coerce")
    result["span_days"] = (
        int((times.max() - times.min()).days) if len(times) > 1 else 0
    )
    return result


def score(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
    from .utils import score_metric

    scores = []
    for key, th in thresholds.items():
        scores.append(score_metric(metrics.get(key), th))
    return sum(scores) / len(scores) if scores else 0.0
