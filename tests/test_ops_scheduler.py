import json
from pathlib import Path

import pytest
import yaml

from alpha.ops.scheduler import ScheduleCfg, run_once, start_scheduler
from alpha.ops.retention import cleanup_runs
import alpha.ops.scheduler as sched_mod
import alpha.ops.notifiers as notifiers

@pytest.fixture
def basic_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    cfg = ScheduleCfg(
        enable=True,
        profile="e2e_h1_m1",
        symbol="EURUSD",
        htf="H1",
        ltf="M1",
        window={"mode": "rolling_days", "days": 1},
        when={"tz": "UTC", "cron": "* * * * *"},
        behavior={"force": False, "resume": True, "fail_fast": True, "dry_run": False},
        backoff={"retries": 2, "initial_s": 0, "max_s": 0, "factor": 1},
        notify={
            "on": ["success"],
            "channels": {"slack": {"enabled": True, "webhook_env": "SLACK_URL"}},
        },
        retention={"runs_to_keep": 5, "delete_old_artifacts": False},
    )
    return cfg


def _prepare_pipeline(monkeypatch, tmp_path, succeed_after=0):
    calls = {"n": 0}

    def fake_run_pipeline(run_cfg):
        calls["n"] += 1
        if calls["n"] <= succeed_after:
            raise RuntimeError("boom")
        run_id = f"20200101_00000{calls['n']}"
        run_dir = Path(run_cfg.io["runs_root"]) / f"{run_cfg.symbol}_{run_cfg.htf}" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text("{}")
        (run_dir / "pipeline.log").write_text("log")
        (run_dir / "trades_summary.json").write_text(
            json.dumps({"n_trades": 4, "win_rate": 0.5, "avg_R": 1.2, "maxDD_R": -1.0})
        )
        reports_dir = Path(run_cfg.io["artifacts_root"]) / "reports" / f"{run_cfg.symbol}_{run_cfg.htf}"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"report_{run_id}.html").write_text("<html></html>")
        return {
            "status": "ok",
            "manifest_path": str(run_dir / "manifest.json"),
            "timings_path": str(run_dir / "timings.json"),
            "log_path": str(run_dir / "pipeline.log"),
        }

    monkeypatch.setattr(sched_mod, "run_pipeline", fake_run_pipeline)
    return calls



def _stub_cli(monkeypatch):
    import types, sys

    m = types.ModuleType("matplotlib")
    m.colors = types.ModuleType("colors")
    sys.modules["matplotlib"] = m
    sys.modules["matplotlib.colors"] = m.colors
    sys.modules["matplotlib.pyplot"] = types.ModuleType("pyplot")
    j2 = types.ModuleType("jinja2")
    j2.Environment = object
    j2.FileSystemLoader = object
    sys.modules["jinja2"] = j2
    from alpha.app.cli import schedule_run
    return schedule_run
def test_schedule_run_once(tmp_path, monkeypatch, basic_cfg):
    schedule_run = _stub_cli(monkeypatch)
    calls = _prepare_pipeline(monkeypatch, tmp_path)
    sent = []
    monkeypatch.setenv("SLACK_URL", "http://example.com")
    monkeypatch.setattr(notifiers, "notify_slack", lambda url, card: sent.append(url))
    monkeypatch.setattr(sched_mod, 'notify_slack', lambda url, card: sent.append(url))

    sched_yaml = tmp_path / "scheduler.yml"
    sched_yaml.write_text(yaml.safe_dump({"schedules": {"job": basic_cfg.__dict__}}))

    schedule_run(profile="e2e_h1_m1", once=True, config_path=str(sched_yaml))
    assert sent and calls["n"] == 1
    sched_root = tmp_path / "artifacts" / "scheduler" / "job"
    assert (sched_root / "runs.jsonl").exists()
    assert (sched_root / "last_status.json").exists()


def test_backoff(tmp_path, monkeypatch, basic_cfg):
    calls = _prepare_pipeline(monkeypatch, tmp_path, succeed_after=2)
    monkeypatch.setenv("SLACK_URL", "http://example.com")
    monkeypatch.setattr(notifiers, "notify_slack", lambda *a, **k: None)
    run_once("job", basic_cfg)
    assert calls["n"] == 3


def test_retention(tmp_path):
    runs_root = tmp_path / "runs"
    for i in range(25):
        (runs_root / f"r{i:02d}").mkdir(parents=True)
    result = cleanup_runs("job", str(runs_root), keep_last=20, delete_artifacts=False)
    assert len(result["kept"]) == 20
    assert len(result["deleted"]) == 5
    assert len(list(runs_root.iterdir())) == 20


def test_missing_env(monkeypatch, tmp_path, basic_cfg, caplog):
    _prepare_pipeline(monkeypatch, tmp_path)
    monkeypatch.delenv("SLACK_URL", raising=False)
    called = []
    monkeypatch.setattr(notifiers, "notify_slack", lambda *a, **k: called.append(1))
    with caplog.at_level("WARNING"):
        run_once("job", basic_cfg)
    assert not called
    assert "missing env" in caplog.text


def test_list_schedules(tmp_path, capfd, basic_cfg):
    schedule_run = _stub_cli(None)
    sched_yaml = tmp_path / "scheduler.yml"
    other = basic_cfg.__dict__.copy()
    other["enable"] = False
    sched_yaml.write_text(
        yaml.safe_dump({"schedules": {"job": basic_cfg.__dict__, "job2": other}})
    )
    schedule_run(list_schedules=True, config_path=str(sched_yaml))
    out = capfd.readouterr().out
    assert "job: enabled" in out
    assert "job2: disabled" in out


def test_start_scheduler(monkeypatch):
    calls = []

    def fake_run_once(sid, cfg):
        calls.append(sid)
        raise KeyboardInterrupt

    monkeypatch.setattr(sched_mod, "run_once", fake_run_once)
    cfg = ScheduleCfg(
        enable=True,
        profile="e2e_h1_m1",
        symbol="EURUSD",
        htf="H1",
        ltf="M1",
        window={"mode": "rolling_days", "days": 1},
        when={"tz": "UTC", "cron": "* * * * *"},
        behavior={},
        backoff={},
        notify={"on": [], "channels": {}},
        retention={},
    )
    start_scheduler({"job": cfg})
    assert calls == ["job"]
