from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def write_summary_json(run_dir: str, data: Dict[str, Any]) -> str:
    path = Path(run_dir) / "summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return str(path)


def write_summary_md(run_dir: str, data: Dict[str, Any]) -> str:
    path = Path(run_dir) / "SUMMARY.md"
    lines = ["| step | status | reason |", "| --- | --- | --- |"]
    for st in data.get("steps", []):
        lines.append(f"| {st['name']} | {st['status']} | {st.get('reason','')} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def save_step_logs(run_dir: str, step: str, stdout: str, stderr: str) -> Dict[str, str]:
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    out_path = rd / f"{step}.stdout.log"
    err_path = rd / f"{step}.stderr.log"
    out_path.write_text(stdout, encoding="utf-8")
    err_path.write_text(stderr, encoding="utf-8")
    return {"stdout": str(out_path), "stderr": str(err_path)}
