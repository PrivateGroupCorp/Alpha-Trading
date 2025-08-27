import json
from pathlib import Path

from alpha.ops.audit import run_repo_audit


def test_repo_audit_smoke(tmp_path: Path):
    outdir = tmp_path / "audit"
    run_repo_audit(Path("."), outdir=outdir, deep=False)
    assert (outdir / "report.json").exists()
    assert (outdir / "REPORT.md").exists()
    assert (outdir / "manifest_files.csv").exists()
    data = json.loads((outdir / "report.json").read_text())
    assert "summary" in data


def test_cli_detection_and_partial_cluster(tmp_path: Path):
    # Build a minimal repository
    root = tmp_path / "repo"
    (root / "alpha" / "app").mkdir(parents=True)
    (root / "alpha" / "structure").mkdir(parents=True)
    cli_code = (
        "import argparse\n"
        "def _build_parser():\n"
        "    p = argparse.ArgumentParser()\n"
        "    sub = p.add_subparsers(dest='command')\n"
        "    sub.add_parser('fetch-data')\n"
        "    return p\n"
    )
    (root / "alpha" / "app" / "cli.py").write_text(cli_code)
    # Create empty file to trigger partial detection
    (root / "alpha" / "structure" / "quality.py").write_text("")

    outdir = root / "artifacts"
    result = run_repo_audit(root=root, outdir=outdir, deep=False)
    data = json.loads((outdir / "report.json").read_text())
    assert result["exit_code"] != 0
    missing_cli = {m["name"] for m in data["executables"]["cli"]["missing"]}
    assert "run-pipeline" in missing_cli
    assert data["clusters"]["structure"]["status"] == "partial"


def test_deep_import_fallback(monkeypatch, tmp_path: Path):
    from alpha.ops import audit as audit_mod

    def bad_import(path: Path):
        raise ImportError("boom")

    monkeypatch.setattr(audit_mod, "import_cli_module", bad_import)
    outdir = tmp_path / "audit"
    res = audit_mod.run_repo_audit(Path("."), outdir=outdir, deep=True)
    assert "import failed" in res.get("notes", "") or res.get("notes")
    assert (outdir / "report.json").exists()
