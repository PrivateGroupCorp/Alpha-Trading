from pathlib import Path
import json
import sys

# Ensure the project root is on sys.path when running via `uv run`
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from alpha.core.data import OHLCFrame
from alpha.core.io import read_mt_tsv, to_parquet, from_parquet
from alpha.core.validate import validate_ohlc_frame
from alpha.app.cli import analyze_levels_data


def test_validate_ohlc_frame_duplicates_and_gaps():
    idx = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:01",
            "2020-01-01 00:01",  # duplicate
            "2020-01-01 00:03",  # gap (missing 00:02)
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [2.0, 2.0, 2.0, 2.0],
            "low": [0.0, 0.0, 0.0, 0.0],
            "close": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )
    ohlc = OHLCFrame(symbol="T", timeframe="M1", tz="UTC", df=df)
    summary = validate_ohlc_frame(ohlc)
    assert summary["duplicate_rows"] == 1
    assert summary["gaps_summary"]["gaps"] == 1
    assert summary["gaps_summary"]["missing_rows"] == 1
    assert len(ohlc.df) == 3  # duplicate removed


def test_read_mt_tsv_roundtrip(tmp_path):
    path = Path("data/EURUSD_H1.tsv")
    ohlc = read_mt_tsv(str(path), symbol="EURUSD", timeframe="H1", tz="UTC")
    validate_ohlc_frame(ohlc)

    out = tmp_path / "ohlc.parquet"
    to_parquet(ohlc, out)
    ohlc2 = from_parquet(out, symbol="EURUSD", timeframe="H1", tz="UTC")
    pd.testing.assert_frame_equal(ohlc.df, ohlc2.df)


def test_cli_analyze_levels_data(tmp_path):
    outdir = tmp_path / "out"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(outdir),
    )
    assert (outdir / "ohlc.parquet").exists()
    assert (outdir / "meta.json").exists()
    meta = json.load(open(outdir / "meta.json"))
    assert meta["symbol"] == "EURUSD"
    assert meta["timeframe"] == "H1"
