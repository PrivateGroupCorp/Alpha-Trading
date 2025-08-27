from __future__ import annotations

"""Utilities for project readiness evaluation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict

from alpha.app.cli import fetch_data_cli
from alpha.qa.health import run_qa
from alpha.qa.utils import load_yaml

from .doctor import run_doctor
from .audit import run_repo_audit
from .audit_catalog import EXPECTED_CONFIGS
from .runner import RunCfg, run_pipeline


@dataclass
class ReadyCfg:
    profile: str
    symbol: str
    htf: str
    ltf: str
    mode: str
    rolling_days: int
    auto_fetch: bool
    provider: str
    force: bool


def run_readiness(cfg: ReadyCfg) -> Dict[str, Any]:
    """Evaluate repository readiness and produce summary artifacts."""

    # Step 1: doctor
    doc_res = run_doctor(profile=cfg.profile, apply=not cfg.force)
    fixes_applied = len(doc_res.get("fixed", []))
    doctor_ok = bool(doc_res.get("ok"))

    # Step 2: repo audit
    audit_res = run_repo_audit()
    summary = audit_res.get("summary", {})
    cli_total = summary.get("cli_total_expected", 0) or 0
    cli_present = summary.get("cli_present", 0) or 0
    cli_cov = cli_present / cli_total if cli_total else 0.0
    cfg_total = len(EXPECTED_CONFIGS)
    cfg_present = summary.get("configs_present", 0) or 0
    cfg_cov = cfg_present / cfg_total if cfg_total else 0.0
    coverage_overall = (cli_cov + cfg_cov) / 2

    # Step 3: ensure data parquet
    parquet_path = Path("data") / cfg.symbol / cfg.htf / "ohlc.parquet"
    data_ok = parquet_path.exists()
    if not data_ok and cfg.auto_fetch:
        end = datetime.utcnow().date()
        start = end - timedelta(days=cfg.rolling_days)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        fetch_data_cli(
            provider=cfg.provider,
            symbol=cfg.symbol,
            tf=cfg.htf,
            start=start.isoformat(),
            end=end.isoformat(),
            outdir=str(parquet_path.parent),
            profile="default",
            tz=None,
            merge=True,
            resample_to=None,
            save_raw=None,
        )
        data_ok = parquet_path.exists()

    # Step 4: pipeline
    behavior = {"dry_run": cfg.mode == "dry"}
    run_cfg = RunCfg(
        profile=cfg.profile,
        symbol=cfg.symbol,
        htf=cfg.htf,
        ltf=cfg.ltf,
        start=None,
        end=None,
        stages={},
        behavior=behavior,
        io={},
    )
    pipe_res = run_pipeline(run_cfg)
    pipeline_status = pipe_res.get("status", "unknown")
    manifest_path = pipe_res.get("manifest_path")
    run_id = Path(manifest_path).parent.name if manifest_path else None

    # Step 5: QA
    qa_cfg = load_yaml("alpha/config/qa.yml").get("qa", {})
    qa_res = run_qa(
        symbol=cfg.symbol,
        tf=cfg.htf,
        run_id=None,
        qa_cfg=qa_cfg,
        artifacts_root="artifacts",
    )
    qa_overall = qa_res.overall
    gates = qa_res.gates

    # Step 6: grade
    if coverage_overall >= 0.8 and pipeline_status == "ok" and qa_overall >= 80:
        grade = "Green"
    elif coverage_overall >= 0.5 and pipeline_status == "ok" and qa_overall >= 50:
        grade = "Yellow"
    else:
        grade = "Red"

    notes = []
    if fixes_applied:
        notes.append(f"doctor_applied={fixes_applied}")
    if not doctor_ok:
        notes.append("doctor_not_ok")
    if not data_ok:
        notes.append("data_missing")
    if pipeline_status != "ok":
        notes.append("pipeline_failed")

    # Step 7: artifacts
    outdir = Path("artifacts") / "readiness" / f"{cfg.symbol}_{cfg.htf}"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "grade": grade,
        "coverage": {
            "cli": cli_cov,
            "config": cfg_cov,
            "overall": coverage_overall,
        },
        "qa_overall": qa_overall,
        "gates": gates,
        "pipeline_status": pipeline_status,
        "run_id": run_id,
        "manifest_path": manifest_path,
        "doctor": {"ok": doctor_ok, "fixes_applied": fixes_applied},
        "data_ok": data_ok,
        "notes": notes,
    }
    json_path = outdir / "readiness.json"
    md_path = outdir / "README.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_lines = [
        f"# Readiness {cfg.symbol} {cfg.htf}",
        "",
        f"Grade: {grade}",
        f"Coverage: {coverage_overall:.2f}",
        f"QA Overall: {qa_overall:.1f}",
        f"Pipeline: {pipeline_status}",
        "",
        "## Gates",
    ]
    for k, v in gates.items():
        md_lines.append(f"- {k}: {'pass' if v else 'fail'}")
    if notes:
        md_lines.append("\n## Notes")
        md_lines.extend(f"- {n}" for n in notes)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    artifacts = {"json": str(json_path), "markdown": str(md_path)}

    return {
        "grade": grade,
        "coverage": result["coverage"],
        "qa_overall": qa_overall,
        "gates": gates,
        "artifacts": artifacts,
        "notes": notes,
    }
