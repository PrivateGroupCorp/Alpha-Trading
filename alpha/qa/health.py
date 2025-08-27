"""Orchestrator for QA health checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from . import checks_data, checks_structure, checks_liquidity, checks_poi, checks_execution, checks_report
from .utils import ensure_dir, save_json, write_badge


@dataclass
class SectionScore:
    name: str
    score: float
    details: Dict[str, Any]


@dataclass
class HealthResult:
    overall: float
    sections: Dict[str, SectionScore]
    gates: Dict[str, Any]
    anomalies_path: Optional[str]
    artifacts: Dict[str, str]
    run_id: Optional[str]


def _evaluate_gates(results: Dict[str, Dict[str, Any]], gates_cfg: Dict[str, Any]) -> Dict[str, bool]:
    gates: Dict[str, bool] = {}
    if gates_cfg.get("require_report"):
        gates["require_report"] = results["report"].get("exists", False)
    min_trades = gates_cfg.get("min_trades")
    if min_trades is not None:
        gates["min_trades"] = results["execution"].get("n_trades", 0) >= min_trades
    min_data_days = gates_cfg.get("min_data_days")
    if min_data_days is not None:
        gates["min_data_days"] = results["data"].get("span_days", 0) >= min_data_days
    return gates


def run_qa(
    symbol: str,
    tf: str,
    run_id: Optional[str],
    qa_cfg: Dict[str, Any],
    artifacts_root: str = "artifacts",
) -> HealthResult:
    root = Path(artifacts_root)
    data_path = root / "data" / symbol / tf
    structure_path = root / "structure" / symbol / tf
    liquidity_path = root / "liquidity" / symbol / tf
    poi_path = root / "poi" / symbol / tf
    execution_path = root / "execution" / symbol / tf
    report_path = root / "report" / symbol / tf

    results: Dict[str, Dict[str, Any]] = {}
    results["data"] = checks_data.run(data_path)
    results["structure"] = checks_structure.run(structure_path)
    results["liquidity"] = checks_liquidity.run(liquidity_path)
    results["poi"] = checks_poi.run(poi_path)
    results["execution"] = checks_execution.run(execution_path)
    results["report"] = checks_report.run(report_path)

    thresholds = qa_cfg.get("thresholds", {})
    sections: Dict[str, SectionScore] = {}
    for name, res in results.items():
        score_fn = getattr(globals()[f"checks_{name}"], "score")
        section_score = score_fn(res, thresholds.get(name, {}))
        sections[name] = SectionScore(name=name, score=section_score, details=res)

    weights = qa_cfg.get("weights", {})
    overall = 0.0
    weight_sum = 0.0
    for name, section in sections.items():
        w = float(weights.get(name, 0.0))
        overall += section.score * w
        weight_sum += w
    if weight_sum:
        overall /= weight_sum

    gates = _evaluate_gates(results, qa_cfg.get("gates", {}))

    outdir = ensure_dir(root / "qa" / f"{symbol}_{tf}" / (run_id or "last"))
    anomalies_path = outdir / "anomalies.csv"
    anomalies_path.write_text("type,detail\n", encoding="utf-8")

    health_json = {
        "overall": overall,
        "sections": {k: {"score": v.score, "details": v.details} for k, v in sections.items()},
        "gates": gates,
        "run_id": run_id,
    }
    save_json(health_json, outdir / "health.json")

    md_lines = [f"# QA Health {symbol} {tf}", "", f"Overall: {overall:.2f}"]
    for name, section in sections.items():
        md_lines.append(f"- {name}: {section.score:.1f}")
    md_lines.append("\n## Gates")
    for k, v in gates.items():
        md_lines.append(f"- {k}: {'pass' if v else 'fail'}")
    (outdir / "HEALTH.md").write_text("\n".join(md_lines), encoding="utf-8")

    badge_path = outdir / "badges" / "health.svg"
    write_badge(overall, badge_path)

    artifacts = {
        "health_json": str(outdir / "health.json"),
        "health_md": str(outdir / "HEALTH.md"),
        "badge": str(badge_path),
        "anomalies": str(anomalies_path),
    }

    return HealthResult(
        overall=overall,
        sections=sections,
        gates=gates,
        anomalies_path=str(anomalies_path),
        artifacts=artifacts,
        run_id=run_id,
    )
