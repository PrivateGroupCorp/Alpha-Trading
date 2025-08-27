import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

def _prepare_cfg(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "alpha" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "qa.yml").write_text("qa: {}\n", encoding="utf-8")


def _fake_pipeline(tmp_path: Path):
    def runner(run_cfg):
        run_id = "20200101_000000"
        run_dir = (
            tmp_path / "runs" / f"{run_cfg.symbol}_{run_cfg.htf}" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        report_dir = (
            tmp_path / "artifacts" / "reports" / f"{run_cfg.symbol}_{run_cfg.htf}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"report_{run_id}.html").write_text("<html></html>")
        return {"status": "ok", "manifest_path": str(manifest_path)}

    return runner


def _fake_qa():
    return types.SimpleNamespace(overall=90, gates={"dummy": True})


def _fake_audit(ok: bool = True):
    if ok:
        return {
            "summary": {
                "cli_total_expected": 1,
                "cli_present": 1,
                "configs_present": 10,
            }
        }
    return {
        "summary": {
            "cli_total_expected": 1,
            "cli_present": 0,
            "configs_present": 0,
        }
    }


@pytest.fixture(autouse=True)
def stub_modules(monkeypatch):
    import types as _types, sys as _sys, argparse

    pd_stub = _types.ModuleType("pandas")
    pd_stub.DataFrame = object
    pd_stub.Series = object
    pd_stub.read_csv = lambda *a, **k: None
    pd_stub.to_datetime = lambda *a, **k: None
    pd_stub.read_parquet = lambda *a, **k: None
    pd_stub.Timestamp = object
    _sys.modules.setdefault("pandas", pd_stub)

    m = _types.ModuleType("matplotlib")
    m.colors = _types.ModuleType("colors")
    _sys.modules["matplotlib"] = m
    _sys.modules["matplotlib.colors"] = m.colors
    _sys.modules["matplotlib.pyplot"] = _types.ModuleType("pyplot")

    j2 = _types.ModuleType("jinja2")
    j2.Environment = object
    j2.FileSystemLoader = object
    _sys.modules["jinja2"] = j2

    yaml_stub = _types.ModuleType("yaml")
    yaml_stub.safe_load = lambda s: {}
    yaml_stub.safe_dump = lambda *a, **k: ""
    _sys.modules.setdefault("yaml", yaml_stub)

    np_stub = _types.ModuleType("numpy")
    np_stub.ndarray = object
    np_stub.array = lambda *a, **k: None
    np_stub.mean = lambda *a, **k: 0
    np_stub.nan = float("nan")
    _sys.modules.setdefault("numpy", np_stub)

    cli_mod = _types.ModuleType("alpha.app.cli")

    def fetch_data_cli(**kwargs):
        pass

    def main():
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("project-readiness")
        p.add_argument("--profile", required=True)
        p.add_argument("--symbol", required=True)
        p.add_argument("--htf", required=True)
        p.add_argument("--ltf", required=True)
        p.add_argument("--mode", default="dry")
        p.add_argument("--rolling-days", type=int, default=30)
        p.add_argument("--auto-fetch", action="store_true")
        p.add_argument("--provider", default="tardis")
        p.add_argument("--force", action="store_true")
        args = parser.parse_args()
        if args.command == "project-readiness":
            from alpha.ops.readiness import ReadyCfg, run_readiness
            cfg = ReadyCfg(
                profile=args.profile,
                symbol=args.symbol,
                htf=args.htf,
                ltf=args.ltf,
                mode=args.mode,
                rolling_days=args.rolling_days,
                auto_fetch=args.auto_fetch,
                provider=args.provider,
                force=args.force,
            )
            result = run_readiness(cfg)
            artifacts = result.get("artifacts", {})
            print(
                f"[project-readiness] grade={result['grade']} artifacts={artifacts}"
            )
            raise SystemExit(1 if result.get("grade") == "Red" else 0)

    cli_mod.fetch_data_cli = fetch_data_cli
    cli_mod.main = main
    _sys.modules.setdefault("alpha.app.cli", cli_mod)


@pytest.mark.usefixtures("tmp_path")
class TestReadinessCLI:
    def test_dry_readiness_smoke(self, tmp_path: Path, monkeypatch, capsys):
        _prepare_cfg(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(REPO_ROOT))

        with patch("alpha.ops.readiness.run_pipeline", side_effect=_fake_pipeline(tmp_path)), \
            patch("alpha.ops.readiness.run_qa", return_value=_fake_qa()), \
            patch("alpha.ops.readiness.fetch_data_cli") as m_fetch, \
            patch("alpha.ops.readiness.run_repo_audit", return_value=_fake_audit()), \
            patch("alpha.ops.readiness.run_doctor", return_value={"ok": True, "fixed": []}):
            from alpha.app import cli as app_cli
            argv = [
                "prog",
                "project-readiness",
                "--profile",
                "p",
                "--symbol",
                "EURUSD",
                "--htf",
                "H1",
                "--ltf",
                "M1",
                "--mode",
                "dry",
            ]
            with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
                app_cli.main()
            assert exc.value.code == 0

        outdir = Path("artifacts") / "readiness" / "EURUSD_H1"
        assert (outdir / "readiness.json").exists()
        assert (outdir / "README.md").exists()
        data = json.loads((outdir / "readiness.json").read_text())
        assert data["grade"] == "Green"
        assert "grade=Green" in capsys.readouterr().out
        assert not m_fetch.called

    def test_smoke_window(self, tmp_path: Path, monkeypatch):
        _prepare_cfg(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(REPO_ROOT))
        with patch("alpha.ops.readiness.run_pipeline", side_effect=_fake_pipeline(tmp_path)), \
            patch("alpha.ops.readiness.run_qa", return_value=_fake_qa()), \
            patch("alpha.ops.readiness.run_repo_audit", return_value=_fake_audit()), \
            patch("alpha.ops.readiness.run_doctor", return_value={"ok": True, "fixed": []}):
            from alpha.app import cli as app_cli
            argv = [
                "prog",
                "project-readiness",
                "--profile",
                "p",
                "--symbol",
                "EURUSD",
                "--htf",
                "H1",
                "--ltf",
                "M1",
                "--mode",
                "smoke",
                "--rolling-days",
                "14",
            ]
            with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
                app_cli.main()
            assert exc.value.code == 0

        outdir = Path("artifacts") / "readiness" / "EURUSD_H1"
        manifest = json.loads((outdir / "readiness.json").read_text())["manifest_path"]
        assert Path(manifest).exists()
        md_text = (outdir / "README.md").read_text()
        assert "../reports/EURUSD_H1/report_20200101_000000.html" in md_text

    def test_exit_codes(self, tmp_path: Path, monkeypatch, capsys):
        _prepare_cfg(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(REPO_ROOT))
        with patch("alpha.ops.readiness.run_pipeline", side_effect=_fake_pipeline(tmp_path)), \
            patch("alpha.ops.readiness.run_qa", return_value=_fake_qa()), \
            patch("alpha.ops.readiness.fetch_data_cli"), \
            patch("alpha.ops.readiness.run_repo_audit", return_value=_fake_audit(ok=False)), \
            patch("alpha.ops.readiness.run_doctor", return_value={"ok": True, "fixed": []}):
            from alpha.app import cli as app_cli
            argv = [
                "prog",
                "project-readiness",
                "--profile",
                "p",
                "--symbol",
                "EURUSD",
                "--htf",
                "H1",
                "--ltf",
                "M1",
                "--mode",
                "dry",
            ]
            with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
                app_cli.main()
            assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "grade=Red" in out

    def test_auto_fetch(self, tmp_path: Path, monkeypatch):
        _prepare_cfg(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(REPO_ROOT))
        parquet_path = Path("data/EURUSD/H1/ohlc.parquet")
        csv_path = parquet_path.with_name("raw.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("d")

        def fake_fetch(**kwargs):
            parquet_path.write_text("p")

        with patch("alpha.ops.readiness.run_pipeline", side_effect=_fake_pipeline(tmp_path)), \
            patch("alpha.ops.readiness.run_qa", return_value=_fake_qa()), \
            patch("alpha.ops.readiness.fetch_data_cli", side_effect=fake_fetch) as m_fetch, \
            patch("alpha.ops.readiness.run_repo_audit", return_value=_fake_audit()), \
            patch("alpha.ops.readiness.run_doctor", return_value={"ok": True, "fixed": []}):
            from alpha.app import cli as app_cli
            argv = [
                "prog",
                "project-readiness",
                "--profile",
                "p",
                "--symbol",
                "EURUSD",
                "--htf",
                "H1",
                "--ltf",
                "M1",
                "--mode",
                "dry",
                "--auto-fetch",
            ]
            with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
                app_cli.main()
        assert m_fetch.called
        assert parquet_path.exists()
