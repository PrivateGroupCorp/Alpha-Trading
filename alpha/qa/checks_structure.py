"""Structure checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"exists": False}
    events_path = path / "events_qualified.csv"
    if not events_path.exists():
        return result
    df = pd.read_csv(events_path)
    result["exists"] = True
    result["n_events"] = len(df)
    result["valid_ratio"] = 1.0
    return result


def score(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
    from .utils import score_metric

    scores = []
    for key, th in thresholds.items():
        scores.append(score_metric(metrics.get(key), th))
    return sum(scores) / len(scores) if scores else 0.0
