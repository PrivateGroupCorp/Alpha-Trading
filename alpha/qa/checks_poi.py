"""POI checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"exists": False}
    zones_path = path / "poi_zones.csv"
    if not zones_path.exists():
        return result
    df = pd.read_csv(zones_path)
    result["exists"] = True
    result["n_zones"] = len(df)
    return result


def score(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
    from .utils import score_metric

    scores = []
    for key, th in thresholds.items():
        scores.append(score_metric(metrics.get(key), th))
    return sum(scores) / len(scores) if scores else 0.0
