from pathlib import Path
from pathlib import Path
import sys
import json

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.levels.detector import LevelsCfgFormation, detect_levels_formation
from alpha.app.cli import analyze_levels_data, analyze_levels_formation


def _bearish_df(n: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2020", periods=n, freq="H", tz="UTC")
    opens = [10 - i for i in range(n)]
    closes = [o - 1 for o in opens]
    highs = [o + 0.5 for o in opens]
    lows = [c - 0.5 for c in closes]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def _bullish_df(n: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2020", periods=n, freq="H", tz="UTC")
    opens = [10 + i for i in range(n)]
    closes = [o + 1 for o in opens]
    highs = [c + 0.5 for c in closes]
    lows = [o - 0.5 for o in opens]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def test_downtrend_produces_peaks():
    df = _bearish_df(4)
    res = detect_levels_formation(df, LevelsCfgFormation())
    assert len(res) == 3
    assert (res["type"] == "peak").all()
    assert res["price"].tolist() == [10.5, 9.5, 8.5]
    assert res["start_idx"].tolist() == [0, 1, 2]
    assert res["end_idx"].tolist() == [1, 2, 3]


def test_uptrend_produces_troughs():
    df = _bullish_df(4)
    res = detect_levels_formation(df, LevelsCfgFormation())
    assert len(res) == 3
    assert (res["type"] == "trough").all()
    assert res["price"].tolist() == [9.5, 10.5, 11.5]


def test_ignore_doji():
    idx = pd.date_range("2021", periods=2, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.02],
            "high": [1.1, 1.3],
            "low": [0.9, 1.0],
            "close": [1.02, 1.2],
        },
        index=idx,
    )
    res = detect_levels_formation(df, LevelsCfgFormation())
    assert res.empty
    res2 = detect_levels_formation(df, LevelsCfgFormation(ignore_doji=False))
    assert len(res2) == 1
    assert res2.iloc[0]["type"] == "trough"
    assert pytest.approx(res2.iloc[0]["price"], rel=1e-6) == 0.9


def test_min_leg_body_ratio():
    idx = pd.date_range("2022", periods=2, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.4],
            "low": [0.8, 1.0],
            "close": [1.05, 1.3],
        },
        index=idx,
    )
    res = detect_levels_formation(df, LevelsCfgFormation(min_leg_body_ratio=0.5, ignore_doji=False))
    assert res.empty


def test_cooldown_bars():
    df = _bearish_df(5)
    cfg = LevelsCfgFormation(cooldown_bars=2)
    res = detect_levels_formation(df, cfg)
    assert len(res) == 2
    assert (res["end_idx"].diff().dropna() >= 2).all()


def test_tick_size_rounding():
    idx = pd.date_range("2023", periods=2, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.4],
            "high": [1.5, 1.8],
            "low": [1.23, 1.26],
            "close": [1.4, 1.7],
        },
        index=idx,
    )
    res = detect_levels_formation(df, LevelsCfgFormation(tick_size=0.1))
    assert len(res) == 1
    assert res.iloc[0]["price"] == pytest.approx(1.2)


def test_integration_levels_formation(tmp_path):
    data_dir = tmp_path / "data"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(data_dir),
    )
    parquet_path = data_dir / "ohlc.parquet"
    outdir = tmp_path / "levels"
    analyze_levels_formation(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
    )

    csv_path = outdir / "levels_formation.csv"
    assert csv_path.exists()
    levels = pd.read_csv(csv_path, parse_dates=["time"])
    assert {"time", "type", "price", "start_idx", "end_idx", "source_closes", "notes"}.issubset(
        levels.columns
    )
    assert len(levels) > 0
    assert levels["type"].isin(["peak", "trough"]).all()
    assert not levels["price"].isna().any()

    summary_path = outdir / "levels_formation_summary.json"
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_bars"] > 0
    assert summary["n_levels"] == len(levels)
    assert summary["density_per_1000"] > 0
