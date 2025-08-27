"""Execution/backtest checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"exists": False}
    trades_path = path / "trades.csv"
    if not trades_path.exists():
        return result
    df = pd.read_csv(trades_path)
    result["exists"] = True
    result["n_trades"] = len(df)
    if "pnl" in df.columns and len(df) > 0:
        result["win_rate"] = (df["pnl"] > 0).mean() * 100.0
    else:
        result["win_rate"] = 0.0
    return result


def score(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
    from .utils import score_metric

    scores = []
    for key, th in thresholds.items():
        scores.append(score_metric(metrics.get(key), th))
    return sum(scores) / len(scores) if scores else 0.0
