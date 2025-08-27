import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "alpha.app.cli", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_import_cli():
    __import__("alpha.app.cli")


def test_help_commands():
    for cmd in ["fetch-data", "run-pipeline", "repo-audit", "project-doctor"]:
        run_cli(cmd, "--help")


def test_project_doctor(tmp_path):
    run_cli("project-doctor")
    doc_dir = ROOT / "artifacts" / "doctor"
    assert (doc_dir / "doctor.log").exists()
    assert (doc_dir / "fixes_applied.json").exists()
    assert (doc_dir / "doctor_summary.md").exists()
