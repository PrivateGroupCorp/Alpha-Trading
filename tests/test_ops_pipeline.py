import json
from pathlib import Path

import yaml

from alpha.ops.runner import RunCfg, run_pipeline


def _load_cfg(tmp_path: Path, behavior_overrides=None):
    cfg_path = Path(__file__).resolve().parents[1] / "alpha" / "config" / "pipeline.yml"
    data = yaml.safe_load(cfg_path.read_text())
    prof = data["profiles"]["e2e_h1_m1"]
    io = prof["io"].copy()
    io["artifacts_root"] = str(tmp_path / "artifacts")
    io["runs_root"] = str(tmp_path / "runs")
    behavior = prof["behavior"].copy()
    if behavior_overrides:
        behavior.update(behavior_overrides)
    run_cfg = RunCfg(
        profile="e2e_h1_m1",
        symbol=prof["symbol"],
        htf=prof["htf"],
        ltf=prof["ltf"],
        start=prof["date"]["start"],
        end=prof["date"]["end"],
        stages=prof["stages"],
        behavior=behavior,
        io=io,
    )
    return run_cfg


def test_pipeline_smoke(tmp_path: Path):
    cfg = _load_cfg(tmp_path)
    result = run_pipeline(cfg)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["data"]["status"] == "disabled"
    for st in ["structure", "liquidity", "poi", "execution", "backtests", "report"]:
        assert manifest[st]["status"] == "ok"
    assert result["status"] == "ok"


def test_pipeline_resume(monkeypatch, tmp_path: Path):
    from alpha.ops import registry as reg

    cfg = _load_cfg(tmp_path)

    original = reg.build_stage_specs

    def failing_build(cfg):
        specs = original(cfg)
        def failing_run(cfg, logger, manifest):
            raise RuntimeError("boom")
        specs["poi"].run_fn = failing_run
        return specs

    monkeypatch.setattr(reg, "build_stage_specs", failing_build)
    res1 = run_pipeline(cfg)
    man1 = json.loads(Path(res1["manifest_path"]).read_text())
    assert man1["poi"]["status"] == "failed"

    monkeypatch.setattr(reg, "build_stage_specs", original)
    cfg.behavior["resume"] = True
    res2 = run_pipeline(cfg)
    man2 = json.loads(Path(res2["manifest_path"]).read_text())
    assert man2["structure"]["status"] == "cached"
    assert man2["poi"]["status"] == "ok"
    assert res2["status"] == "ok"


def test_pipeline_force_vs_cached(tmp_path: Path):
    cfg = _load_cfg(tmp_path)
    run_pipeline(cfg)
    res2 = run_pipeline(cfg)
    man2 = json.loads(Path(res2["manifest_path"]).read_text())
    assert man2["structure"]["status"] == "cached"
    cfg.behavior["force"] = True
    res3 = run_pipeline(cfg)
    man3 = json.loads(Path(res3["manifest_path"]).read_text())
    assert man3["structure"]["status"] == "ok"


def test_pipeline_dry_run(tmp_path: Path):
    cfg = _load_cfg(tmp_path, behavior_overrides={"dry_run": True})
    res = run_pipeline(cfg)
    man = json.loads(Path(res["manifest_path"]).read_text())
    for st in ["structure", "liquidity", "poi", "execution", "backtests", "report"]:
        assert man[st]["status"] == "dry_run"
        out_path = Path(cfg.io["artifacts_root"]) / st / f"{cfg.symbol}_{cfg.htf}" / f"{st}.txt"
        if st == "report":
            out_path = Path(cfg.io["artifacts_root"]) / "reports" / f"{cfg.symbol}_{cfg.htf}" / "report.html"
        assert not out_path.exists()
