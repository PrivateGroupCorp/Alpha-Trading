"""Utility helpers for QA health checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file and return dictionary (empty if missing)."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save dictionary as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def score_less_is_better(value: float | None, max_value: float) -> float:
    if value is None:
        return 0.0
    if value <= 0:
        return 100.0
    if value >= max_value:
        return 0.0
    return max(0.0, 100.0 * (1 - value / max_value))


def score_within_range(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if low <= value <= high:
        return 100.0
    if value < low:
        diff = low - value
        span = abs(low) if low != 0 else 1.0
        return max(0.0, 100.0 - 100.0 * diff / span)
    diff = value - high
    span = abs(high) if high != 0 else 1.0
    return max(0.0, 100.0 - 100.0 * diff / span)


def score_metric(value: float | None, threshold: Any) -> float:
    if isinstance(threshold, (int, float)):
        return score_less_is_better(value, float(threshold))
    if (
        isinstance(threshold, (list, tuple))
        and len(threshold) == 2
        and all(isinstance(t, (int, float)) for t in threshold)
    ):
        return score_within_range(value, float(threshold[0]), float(threshold[1]))
    return 0.0


def write_badge(score: float, path: str | Path) -> None:
    """Write a very small SVG badge indicating score severity."""
    color = "#e43"  # red
    if score >= 80:
        color = "#3c1"  # green
    elif score >= 50:
        color = "#fc3"  # yellow
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='20'>"
        "<rect width='100' height='20' fill='%s'/>" % color +
        f"<text x='50' y='14' font-size='12' text-anchor='middle' fill='black'>{score:.1f}</text>" +
        "</svg>"
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(svg, encoding="utf-8")
