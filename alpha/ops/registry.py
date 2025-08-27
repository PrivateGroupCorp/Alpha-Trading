"""Stage registry for pipeline runner.

This module provides utilities to build a mapping of :class:`StageSpec`
objects for the pipeline runner.  The default implementation used in the
tests is intentionally lightweight – each stage simply creates a small file
in the artifacts directory to simulate work being done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .stages import StageSpec

try:  # pragma: no cover - circular import guard
    from .runner import RunCfg
except Exception:  # pragma: no cover
    RunCfg = object  # type: ignore

STAGE_ORDER = [
    "data",
    "structure",
    "liquidity",
    "poi",
    "execution",
    "backtests",
    "report",
]

DEPS = {
    "data": [],
    "structure": ["data"],
    "liquidity": ["data"],
    "poi": ["structure", "liquidity"],
    "execution": ["poi"],
    "backtests": ["poi"],
    "report": ["execution", "backtests"],
}


def _output_path(cfg: RunCfg, stage: str) -> Path:
    root = Path(cfg.io.get("artifacts_root", "artifacts"))
    token = f"{cfg.symbol}_{cfg.htf}"
    if stage == "report":
        return root / "reports" / token / "report.html"
    if stage == "data":
        return root / "data" / cfg.symbol / cfg.ltf / "ohlc.parquet"
    # default location
    return root / stage / token / f"{stage}.txt"


def _make_run_fn(stage: str, out_path: Path):
    def run_fn(cfg: RunCfg, logger, manifest):  # pragma: no cover - thin wrapper
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("ok")
        return {"outputs": [str(out_path)], "ok": True}

    return run_fn


def _make_health_fn(out_path: Path):
    def health_fn(cfg: RunCfg):
        ok = out_path.exists()
        return {"ok": ok, "outputs": [str(out_path)]}

    return health_fn


def build_stage_specs(cfg: RunCfg) -> Dict[str, StageSpec]:
    """Build StageSpec objects for all known stages."""

    specs: Dict[str, StageSpec] = {}
    for name in STAGE_ORDER:
        stage_cfg = cfg.stages.get(name, {}) if cfg.stages else {}
        enabled = bool(stage_cfg.get("enabled", True))
        out_path = _output_path(cfg, name)
        spec = StageSpec(
            name=name,
            enabled=enabled,
            deps=DEPS.get(name, []),
            expected_outputs=[str(out_path)],
            run_fn=_make_run_fn(name, out_path),
            healthcheck_fn=_make_health_fn(out_path),
        )
        specs[name] = spec
    return specs
