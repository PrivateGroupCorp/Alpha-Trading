from __future__ import annotations

"""Repository audit utility.

This module scans the repository and produces a report about implemented
modules, configuration files, tests and CLI entry points.  Results are written
as JSON, Markdown and CSV manifests.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import importlib.util

from .audit_catalog import (
    EXPECTED_CLI_COMMANDS,
    TASK_FILE_MAP,
    EXPECTED_CONFIGS,
    EXPECTED_TESTS,
    CRITICAL_CLI,
)


def import_cli_module(path: Path):
    """Import ``alpha/app/cli.py`` from an arbitrary path.

    Using importlib util to avoid depending on package installation.
    """

    spec = importlib.util.spec_from_file_location("repo_cli", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("cannot create spec for cli module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _discover_cli(path: Path, expected: List[str], deep: bool) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Discover CLI commands.

    Returns tuple (present, missing, notes).
    """

    present: List[Dict] = []
    missing: List[Dict] = []
    notes: List[str] = []

    if not path.exists():
        for name in expected:
            missing.append({"name": name, "expected_in": str(path)})
        return present, missing, notes

    if deep:
        try:
            module = import_cli_module(path)
            if hasattr(module, "_build_parser"):
                parser = module._build_parser()
                names = set()
                for action in parser._actions:
                    if isinstance(action, argparse._SubParsersAction):
                        names.update(action.choices.keys())
                for name in expected:
                    if name in names:
                        present.append({"name": name, "path": str(path), "via": "import"})
                    else:
                        missing.append({"name": name, "expected_in": str(path)})
                return present, missing, notes
            else:  # pragma: no cover - unlikely
                notes.append("_build_parser missing in cli module")
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"import failed: {exc}")
            present.clear()
            missing.clear()
            # fall back to regex below

    # Fallback: simple text search
    text = path.read_text(encoding="utf-8")
    for name in expected:
        if name in text:
            present.append({"name": name, "path": str(path), "via": "regex"})
        else:
            missing.append({"name": name, "expected_in": str(path)})
    return present, missing, notes


def _file_status(p: Path) -> str:
    if p.exists() and p.is_file() and p.stat().st_size > 0:
        return "present"
    elif p.exists() and p.is_file():
        return "partial"
    return "missing"


def run_repo_audit(root: Path | str = Path("."), outdir: Path | str = Path("artifacts/audit"), *, deep: bool = False) -> Dict:
    """Run repository audit.

    Parameters
    ----------
    root: Path or str
        Root of the repository to scan.
    outdir: Path or str
        Directory where reports should be written.
    deep: bool
        If ``True`` attempt to import the CLI module to discover commands.

    Returns
    -------
    Dict
        Report dictionary including ``exit_code`` key.
    """

    root = Path(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []

    # Discover CLI commands
    cli_path = root / "alpha" / "app" / "cli.py"
    cli_present, cli_missing, cli_notes = _discover_cli(cli_path, EXPECTED_CLI_COMMANDS, deep)
    manifest_rows.append({"path": str(cli_path), "status": _file_status(cli_path)})

    # Evaluate tasks / clusters
    clusters: Dict[str, Dict] = {}
    tasks_total = 0
    tasks_ok = 0
    for cl_name, tasks in TASK_FILE_MAP.items():
        cl_tasks: Dict[str, Dict] = {}
        cl_statuses: List[str] = []
        for task_code, files in tasks.items():
            tasks_total += 1
            file_statuses = []
            for f in files:
                p = root / f
                st = _file_status(p)
                manifest_rows.append({"path": f, "status": st})
                file_statuses.append(st)
            if all(st == "present" for st in file_statuses):
                status = "implemented"
                tasks_ok += 1
            elif any(st != "missing" for st in file_statuses):
                status = "partial"
            else:
                status = "missing"
            cl_tasks[task_code] = {"files": files, "status": status}
            cl_statuses.append(status)
        if all(st == "implemented" for st in cl_statuses):
            cl_status = "implemented"
        elif any(st != "missing" for st in cl_statuses):
            cl_status = "partial"
        else:
            cl_status = "missing"
        clusters[cl_name] = {"status": cl_status, "tasks": cl_tasks}

    # Config files
    cfg_present: List[str] = []
    cfg_missing: List[str] = []
    for cfg in EXPECTED_CONFIGS:
        p = root / cfg
        st = _file_status(p)
        manifest_rows.append({"path": cfg, "status": st})
        if st == "present":
            cfg_present.append(cfg)
        else:
            cfg_missing.append(cfg)

    # Tests
    tests_present: List[str] = []
    tests_missing: List[str] = []
    for tst in EXPECTED_TESTS:
        p = root / tst
        st = _file_status(p)
        manifest_rows.append({"path": tst, "status": st})
        if st == "present":
            tests_present.append(tst)
        else:
            tests_missing.append(tst)

    # Summary and grade
    clusters_total = len(TASK_FILE_MAP)
    clusters_ok = sum(1 for c in clusters.values() if c["status"] == "implemented")
    cli_total = len(EXPECTED_CLI_COMMANDS)
    cli_present_count = len(cli_present)
    summary = {
        "clusters_total": clusters_total,
        "clusters_ok": clusters_ok,
        "tasks_total": tasks_total,
        "tasks_ok": tasks_ok,
        "cli_total_expected": cli_total,
        "cli_present": cli_present_count,
        "configs_present": len(cfg_present),
        "tests_present": len(tests_present),
    }

    coverage = 0.0
    if tasks_total:
        coverage += tasks_ok / tasks_total
    if cli_total:
        coverage += cli_present_count / cli_total
    coverage /= 2 if coverage else 1
    grade = "A" if coverage >= 0.8 else "B" if coverage >= 0.5 else "C"

    executables = {
        "cli": {"present": cli_present, "missing": cli_missing},
        "scripts": {"present": [], "missing": []},
    }
    configs = {"present": cfg_present, "missing": cfg_missing}
    tests = {"present": tests_present, "missing": tests_missing}

    notes = "; ".join(cli_notes) if cli_notes else ""

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo_root": str(root.resolve()),
        "summary": summary,
        "executables": executables,
        "clusters": clusters,
        "configs": configs,
        "tests": tests,
        "notes": notes,
        "grade": grade,
    }

    # Write outputs
    report_json = outdir / "report.json"
    report_md = outdir / "REPORT.md"
    manifest_csv = outdir / "manifest_files.csv"
    with report_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_report_md(report_md, clusters, cli_present, cfg_present, cfg_missing, tests_present, tests_missing)
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "status"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    # Console output
    print(
        f"[repo-audit] clusters: ok={tasks_ok}/{tasks_total} tasks, cli={cli_present_count}/{cli_total} present, configs={len(cfg_present)}/{len(EXPECTED_CONFIGS)}, tests={len(tests_present)}/{len(EXPECTED_TESTS)}"
    )
    print(f"[repo-audit] report: {report_json}")
    print(f"[repo-audit] summary: {report_md}")

    # Critical CLI missing?
    cli_present_names = {c["name"] for c in cli_present}
    critical_missing = [c for c in CRITICAL_CLI if c not in cli_present_names]
    if critical_missing:
        print(f"[repo-audit] MISSING CRITICAL: {', '.join(critical_missing)}")
    exit_code = 1 if critical_missing else 0
    report["exit_code"] = exit_code
    return report


def _write_report_md(
    path: Path,
    clusters: Dict[str, Dict],
    cli_present: List[Dict],
    cfg_present: List[str],
    cfg_missing: List[str],
    tests_present: List[str],
    tests_missing: List[str],
) -> None:
    """Write a human readable markdown summary."""

    lines: List[str] = []
    lines.append("# Repo Audit")
    lines.append("")
    lines.append("## Clusters")
    lines.append("| Cluster | Status |")
    lines.append("| --- | --- |")
    for cl_name, cl in clusters.items():
        lines.append(f"| {cl_name} | {cl['status']} |")
    lines.append("")
    lines.append("## CLI Commands")
    lines.append("| Command | Status |")
    lines.append("| --- | --- |")
    present_names = {c["name"] for c in cli_present}
    for name in EXPECTED_CLI_COMMANDS:
        status = "present" if name in present_names else "missing"
        lines.append(f"| {name} | {status} |")
    lines.append("")
    lines.append("## Configs")
    lines.append("### Present")
    for cfg in cfg_present:
        lines.append(f"- {cfg}")
    if cfg_missing:
        lines.append("### Missing")
        for cfg in cfg_missing:
            lines.append(f"- {cfg}")
    lines.append("")
    lines.append("## Tests")
    lines.append("### Present")
    for t in tests_present:
        lines.append(f"- {t}")
    if tests_missing:
        lines.append("### Missing")
        for t in tests_missing:
            lines.append(f"- {t}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

