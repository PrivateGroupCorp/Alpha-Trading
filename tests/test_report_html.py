import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.app.cli import generate_report


def _create_png(path: Path) -> None:
    plt.figure()
    plt.plot([1, 2], [1, 2])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _prepare_artifacts(base: Path) -> None:
    token = "EURUSD_H1"
    struct_dir = base / "structure" / token
    struct_dir.mkdir(parents=True, exist_ok=True)
    _create_png(struct_dir / "structure_last500.png")
    (struct_dir / "trend_summary.json").write_text(
        json.dumps({
            "share_time_up": 0.4,
            "share_time_down": 0.3,
            "share_time_range": 0.3,
            "n_reversals": 5,
        })
    )

    poi_dir = base / "poi" / token
    poi_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "kind": ["A", "B"],
            "grade": [1, 2],
            "width_pips": [5.0, 6.0],
            "score_total": [10, 8],
        }
    ).to_csv(poi_dir / "poi_zones.csv", index=False)

    liq_dir = base / "liquidity" / token
    liq_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "width_pips": [10.0, 12.0],
            "break_up": [1, 0],
        }
    ).to_csv(liq_dir / "asia_range_daily.csv", index=False)

    exe_dir = base / "execution" / token
    exe_dir.mkdir(parents=True, exist_ok=True)
    (exe_dir / "trades_summary.json").write_text(
        json.dumps({"n_trades": 2, "win_rate": 0.5, "avg_R": 1.2, "maxDD_R": -0.5})
    )


def test_report_html_smoke(tmp_path: Path):
    base = tmp_path / "artifacts"
    _prepare_artifacts(base)

    outdir = base / "reports" / "EURUSD_H1"
    generate_report(symbol="EURUSD", tf="H1", outdir=str(outdir))

    html_files = list(outdir.glob("report_*.html"))
    assert html_files, "HTML report not generated"
    html_text = html_files[0].read_text()
    assert "Alpha-POI Report" in html_text
    assert "<img" in html_text
    assert "<table" in html_text
    assert "Download CSV" in html_text

    snap = json.loads((outdir / "params_snapshot.json").read_text())
    assert "structure.yml" in snap


def test_report_html_missing_sections(tmp_path: Path):
    base = tmp_path / "artifacts"
    _prepare_artifacts(base)
    # remove execution to simulate missing
    import shutil
    shutil.rmtree(base / "execution")

    outdir = base / "reports" / "EURUSD_H1"
    generate_report(symbol="EURUSD", tf="H1", outdir=str(outdir))
    html_files = list(outdir.glob("report_*.html"))
    assert html_files
    html_text = html_files[0].read_text()
    # trades section should be absent
    assert "n_trades" not in html_text


def test_number_formatting(tmp_path: Path):
    base = tmp_path / "artifacts"
    _prepare_artifacts(base)

    outdir = base / "reports" / "EURUSD_H1"
    generate_report(symbol="EURUSD", tf="H1", outdir=str(outdir))
    html_files = list(outdir.glob("report_*.html"))
    html_text = html_files[0].read_text()
    # width_pips should be formatted with one decimal place
    assert re.search(r"10\.0", html_text)
