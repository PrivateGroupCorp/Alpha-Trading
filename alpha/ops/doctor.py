"""Project doctor: audit → fix → verify utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .audit import run_repo_audit
from .runner import RunCfg, run_pipeline
from .doctor_catalog import REQUIRED_FILES, REQUIREMENTS


@dataclass
class FixAction:
    kind: str
    target: str
    detail: Dict[str, Any]


def _ensure_file(path: Path, content: str) -> FixAction | None:
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return FixAction("write_file", str(path), {"created": True})


def _ensure_requirements(path: Path) -> FixAction | None:
    existing: List[str] = []
    if path.exists():
        existing = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    merged = sorted(set(existing) | set(REQUIREMENTS))
    if existing != merged:
        path.write_text("\n".join(merged) + "\n", encoding="utf-8")
        return FixAction("ensure_requirements", str(path), {"count": len(merged)})
    return None


def run_doctor(root: str = ".", profile: str = "e2e_h1_m1", apply: bool = True, dry_run: bool = False) -> Dict[str, Any]:
    """Run audit, apply fixes and verify pipeline readiness."""

    root_path = Path(root).resolve()
    artifacts_dir = root_path / "artifacts" / "doctor"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_path = artifacts_dir / "doctor.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("alpha.ops.doctor")
    logger.info("doctor start")

    fixes: List[FixAction] = []
    remaining: List[str] = []

    try:
        run_repo_audit(root=str(root_path), outdir=str(root_path / "artifacts" / "audit"))
    except Exception as exc:  # pragma: no cover - defensive
        remaining.append(f"audit_pre: {exc}")

    if apply:
        for rel, content in REQUIRED_FILES.items():
            action = _ensure_file(root_path / rel, content)
            if action:
                fixes.append(action)
        req_action = _ensure_requirements(root_path / "requirements.txt")
        if req_action:
            fixes.append(req_action)

    try:
        run_repo_audit(root=str(root_path), outdir=str(root_path / "artifacts" / "audit"))
    except Exception as exc:  # pragma: no cover - defensive
        remaining.append(f"audit_post: {exc}")

    if not dry_run:
        try:
            cfg_path = root_path / "alpha" / "config" / "pipeline.yml"
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            prof = data.get("profiles", {}).get(profile, {})
            date_cfg = prof.get("date", {})
            run_cfg = RunCfg(
                profile=profile,
                symbol=prof.get("symbol"),
                htf=prof.get("htf"),
                ltf=prof.get("ltf"),
                start=date_cfg.get("start"),
                end=date_cfg.get("end"),
                stages=prof.get("stages", {}),
                behavior={**prof.get("behavior", {}), "dry_run": True},
                io=prof.get("io", {}),
            )
            run_pipeline(run_cfg)
        except Exception as exc:  # pragma: no cover - defensive
            remaining.append(f"pipeline: {exc}")

    fixes_path = artifacts_dir / "fixes_applied.json"
    fixes_path.write_text(
        json.dumps([f.__dict__ for f in fixes], indent=2), encoding="utf-8"
    )
    summary_path = artifacts_dir / "doctor_summary.md"
    lines = ["# Project Doctor Summary", ""]
    lines.append(f"Total fixes applied: {len(fixes)}")
    if remaining:
        lines.append("\n## Remaining Issues")
        lines.extend(f"- {r}" for r in remaining)
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("doctor end")
    return {
        "ok": not remaining,
        "fixed": fixes,
        "remaining_issues": remaining,
        "audit_report": str(root_path / "artifacts" / "audit" / "report.json"),
    }
