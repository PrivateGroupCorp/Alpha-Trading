import json
import sys
from pathlib import Path

import pytest

from alpha.ops.run_all import RunAllCfg, run_all
from alpha.ops import steps, install


def _cfg():
    return RunAllCfg(
        profile="p",
        symbol="EURUSD",
        htf="H1",
        ltf="M1",
        mode="dry",
        rolling_days=1,
        install_missing=False,
        auto_fetch=False,
        provider="csv",
        continue_on_error=True,
        max_retries=1,
        timeout_per_step_sec=10,
        run_bt=False,
        skip_backtests_vbt=False,
    )


def _all_pass_funcs():
    funcs = {}
    for name in steps.STEP_ORDER:
        funcs[name] = lambda cfg, name=name: steps.StepResult(
            name, "PASS", 0.0
        )
    return funcs


def test_dry_green_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg()
    monkeypatch.setattr(steps, "STEP_FUNCS", _all_pass_funcs())
    res = run_all(cfg)
    assert res["grade"] == "GREEN"
    assert res["exit_code"] == 0
    assert Path(res["summary_json"]).exists()


def test_missing_pkg_autoinstall(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg()
    cfg.install_missing = True

    funcs = _all_pass_funcs()
    calls = {"n": 0}

    def pipeline(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("No module named 'pkgx'")
        return steps.StepResult("pipeline_dry_or_smoke", "PASS", 0.0)

    funcs["pipeline_dry_or_smoke"] = pipeline
    monkeypatch.setattr(steps, "STEP_FUNCS", funcs)
    monkeypatch.setattr(install, "try_autoinstall_from_error", lambda s: "pkgx")
    res = run_all(cfg)
    step = next(s for s in res["steps"] if s["name"] == "pipeline_dry_or_smoke")
    assert step["status"] == "PASS"
    assert calls["n"] == 2


def test_data_missing_skip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg()
    funcs = _all_pass_funcs()
    funcs["data_bootstrap"] = lambda cfg: steps.StepResult(
        "data_bootstrap", "SKIP", 0.0, reason="data_missing"
    )
    monkeypatch.setattr(steps, "STEP_FUNCS", funcs)
    res = run_all(cfg)
    step = next(s for s in res["steps"] if s["name"] == "data_bootstrap")
    assert step["status"] == "SKIP"
    assert step["reason"] == "data_missing"


def test_pytest_fail_exit_yellow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg()
    funcs = _all_pass_funcs()
    funcs["pytest"] = lambda cfg: steps.StepResult(
        "pytest", "FAIL", 0.0, reason="boom"
    )
    monkeypatch.setattr(steps, "STEP_FUNCS", funcs)
    res = run_all(cfg)
    assert res["grade"] == "YELLOW"
    assert res["exit_code"] == 0


def test_red_on_cli_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    funcs = _all_pass_funcs()
    funcs["repo_audit"] = lambda cfg: steps.StepResult(
        "repo_audit", "FAIL", 0.0, reason="cli_missing"
    )
    monkeypatch.setattr(steps, "STEP_FUNCS", funcs)

    import types, importlib

    fake = types.ModuleType("pydantic")
    fake.BaseModel = object
    fake.ConfigDict = dict
    sys.modules.setdefault("pydantic", fake)

    from alpha.app import cli as cli_module

    argv = [
        "alpha-cli",
        "project-run-all",
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
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as se:
        cli_module.main()
    assert se.value.code == 1
