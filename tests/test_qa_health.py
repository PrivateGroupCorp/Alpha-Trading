import subprocess
import sys
from pathlib import Path

import pandas as pd

from alpha.qa.utils import score_metric
from alpha.qa.health import run_qa


def _prepare_artifacts(root: Path) -> None:
    data_dir = root / "data" / "EURUSD" / "H1"
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "time": dates,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
    })
    df.loc[0, "time"] = df.loc[1, "time"]  # introduce 1 duplicate -> 1% dup
    df.to_parquet(data_dir / "ohlc.parquet", index=False)

    exec_dir = root / "execution" / "EURUSD" / "H1"
    exec_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame({"pnl": [1] * 10})
    trades.to_csv(exec_dir / "trades.csv", index=False)


def test_score_mapping():
    assert score_metric(0.0, 10) == 100
    assert score_metric(5.0, 10) == 50
    assert score_metric(10.0, 10) == 0


def test_gate_require_report(tmp_path):
    _prepare_artifacts(tmp_path)
    cmd = [
        sys.executable,
        "-m",
        "alpha.app.cli",
        "qa-run",
        "--symbol",
        "EURUSD",
        "--tf",
        "H1",
        "--artifacts-root",
        str(tmp_path),
        "--strict-mode",
        "hard",
    ]
    res = subprocess.run(cmd, capture_output=True)
    assert res.returncode != 0


def test_overall_weighted(tmp_path):
    _prepare_artifacts(tmp_path)
    report_dir = tmp_path / "report" / "EURUSD" / "H1"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.html").write_text("hi")

    qa_cfg = {
        "weights": {"data": 0.5, "report": 0.5},
        "gates": {},
        "thresholds": {
            "data": {"dup_pct": 2},
            "report": {"size_kb": 1},
        },
    }
    result = run_qa("EURUSD", "H1", None, qa_cfg, artifacts_root=str(tmp_path))
    assert result.sections["data"].score < 100
    assert result.sections["report"].score > 99
    expected = 0.5 * result.sections["data"].score + 0.5 * result.sections["report"].score
    assert abs(result.overall - expected) < 1e-6
