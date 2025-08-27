"""Report checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def run(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"exists": False}
    html_files = list(path.glob("report*.html"))
    if not html_files:
        return result
    report = html_files[0]
    result["exists"] = True
    result["size_kb"] = report.stat().st_size / 1024.0
    result["path"] = str(report)
    return result


def score(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
    from .utils import score_metric

    scores = []
    for key, th in thresholds.items():
        scores.append(score_metric(metrics.get(key), th))
    return sum(scores) / len(scores) if scores else 0.0
