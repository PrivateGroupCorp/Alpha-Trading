import pandas as pd
from pathlib import Path

from alpha.data.normalize import normalize_ohlc, NormCfg
from alpha.data.resample import resample_ohlc
from alpha.data.merge import merge_with_existing
from alpha.data.validators import quality_report
from alpha.data.ingest import fetch_and_normalize, save_pipeline


def test_normalize_ohlc():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:01:00",
                "2024-01-01 00:00:00",
                "2024-01-01 00:01:00",
            ],
            "Open": [1.5, 1.0, 1.6],
            "High": [2.0, 1.5, 2.1],
            "Low": [0.9, 0.5, 1.0],
            "Close": [1.7, 1.4, 1.8],
            "Volume": [100, 50, 120],
        }
    ).set_index("timestamp")
    cfg = NormCfg()
    out = normalize_ohlc(df, cfg)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing
    assert len(out) == 2  # duplicate removed


def test_resample_ohlc():
    idx = pd.date_range("2024-01-01", periods=30, freq="T", tz="UTC")
    df = pd.DataFrame(
        {
            "open": range(30),
            "high": range(30),
            "low": range(30),
            "close": range(30),
            "volume": [1] * 30,
        },
        index=idx,
    )
    res = resample_ohlc(df, "M15")
    assert len(res) == 2
    assert res.iloc[0]["open"] == 0
    assert res.iloc[0]["close"] == 14
    assert res.iloc[0]["volume"] == 15


def test_merge_with_existing(tmp_path):
    idx1 = pd.date_range("2024-01-01", periods=2, freq="T", tz="UTC")
    df1 = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2]}, index=idx1)
    path = tmp_path / "ohlc.parquet"
    df1.to_parquet(path)
    idx2 = pd.date_range("2024-01-01 00:01", periods=2, freq="T", tz="UTC")
    df2 = pd.DataFrame({"open": [2, 3], "high": [2, 3], "low": [2, 3], "close": [2, 3]}, index=idx2)
    merged = merge_with_existing(df2, path)
    assert len(merged) == 3
    assert merged.index.is_monotonic_increasing


def test_quality_report():
    idx = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:03", "2024-01-01 00:03"],
        utc=True,
    )
    df = pd.DataFrame({"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]}, index=idx)
    rep = quality_report(df, "M1")
    assert rep["gaps"]["count"] == 1
    assert rep["duplicates_removed"] == 1


def test_csv_local_integration(tmp_path):
    result = fetch_and_normalize(
        provider_name="csv_local",
        symbol="EURUSD",
        tf="M1",
        start="2024-01-01 00:00:00",
        end="2024-01-01 00:02:00",
        profile="default",
    )
    outdir = tmp_path / "EURUSD" / "M1"
    save_pipeline(
        result["df"],
        symbol="EURUSD",
        tf="M1",
        outdir=str(outdir),
        merge=True,
        resample_to=["H1"],
        report=result["report"],
    )
    assert (outdir / "ohlc.parquet").exists()
    assert (outdir / "quality_report.json").exists()
    assert (outdir.parent / "H1" / "ohlc.parquet").exists()
