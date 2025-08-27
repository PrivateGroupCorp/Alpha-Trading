"""Simple pipeline runner orchestrating stage execution."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .stages import StageSpec
import alpha.ops.registry as registry


@dataclass
class RunCfg:
    profile: str
    symbol: str
    htf: str
    ltf: str
    start: str | None
    end: str | None
    stages: Dict[str, Any]
    behavior: Dict[str, Any]
    io: Dict[str, Any]


def _topo_sort(specs: Dict[str, StageSpec]) -> List[StageSpec]:
    remaining = dict(specs)
    sorted_specs: List[StageSpec] = []
    satisfied: set[str] = set()
    while remaining:
        ready = [name for name, sp in remaining.items() if all(d in satisfied for d in sp.deps)]
        if not ready:
            raise ValueError("Circular dependency in stage definitions")
        for name in ready:
            sp = remaining.pop(name)
            sorted_specs.append(sp)
            satisfied.add(name)
    return sorted_specs


def run_pipeline(cfg: RunCfg) -> Dict[str, Any]:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    runs_root = Path(cfg.io.get("runs_root", "artifacts/runs"))
    run_dir = runs_root / f"{cfg.symbol}_{cfg.htf}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "pipeline.log"
    logger = logging.getLogger("alpha.ops.runner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    specs = registry.build_stage_specs(cfg)
    order = _topo_sort(specs)

    manifest: Dict[str, Any] = {}
    timings: Dict[str, float] = {}
    overall_status = "ok"

    for spec in order:
        t0 = time.monotonic()
        logger.info("stage %s start", spec.name)
        if not spec.enabled:
            manifest[spec.name] = {"status": "disabled"}
            logger.info("stage %s disabled", spec.name)
            continue

        outputs_exist = all(Path(p).exists() for p in spec.expected_outputs)
        if outputs_exist and not cfg.behavior.get("force", False):
            manifest[spec.name] = {"status": "cached", "outputs": spec.expected_outputs}
            logger.info("stage %s cached", spec.name)
        elif cfg.behavior.get("dry_run", False):
            hc = spec.healthcheck_fn(cfg) if spec.healthcheck_fn else {"ok": True}
            manifest[spec.name] = {"status": "dry_run", "health": hc}
            logger.info("stage %s dry-run", spec.name)
        else:
            try:
                result = spec.run_fn(cfg, logger, manifest) if spec.run_fn else {"ok": True}
                hc = spec.healthcheck_fn(cfg) if spec.healthcheck_fn else {"ok": True}
                ok = result.get("ok", False) and hc.get("ok", False)
                status = "ok" if ok else "failed"
                manifest[spec.name] = {
                    "status": status,
                    "outputs": result.get("outputs", []),
                    "notes": result.get("notes", ""),
                }
                if not ok:
                    overall_status = "failed"
                    logger.error("stage %s failed", spec.name)
                    if cfg.behavior.get("fail_fast", True):
                        break
            except Exception as exc:  # pragma: no cover - defensive
                overall_status = "failed"
                manifest[spec.name] = {"status": "failed", "error": str(exc)}
                logger.exception("stage %s raised exception", spec.name)
                if cfg.behavior.get("fail_fast", True):
                    break

        dt = time.monotonic() - t0
        timings[spec.name] = dt
        logger.info("stage %s end %.2fs", spec.name, dt)

    manifest_path = run_dir / "manifest.json"
    timings_path = run_dir / "timings.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with timings_path.open("w", encoding="utf-8") as fh:
        json.dump(timings, fh, indent=2)

    return {
        "status": overall_status,
        "manifest_path": str(manifest_path),
        "timings_path": str(timings_path),
        "log_path": str(log_path),
    }
