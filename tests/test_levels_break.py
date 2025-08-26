from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.core.indicators import atr
from alpha.levels.break_update import LevelsCfgBreakUpdate, apply_break_update
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_levels_break,
)


def test_break_body_peak():
    idx = pd.date_range("2020", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [0.5, 0.5],
            "high": [1.0, 1.5],
            "low": [0.0, 0.0],
            "close": [0.5, 1.2],
        },
        index=idx,
    )
    levels = pd.DataFrame(
        [{"time": idx[0], "type": "peak", "price": 0.5, "start_idx": 0, "end_idx": 0}]
    )
    cfg = LevelsCfgBreakUpdate(break_mode="atr", atr_mult=0.5)
    res = apply_break_update(df, levels, cfg)
    assert res.loc[0, "state"] == "broken"
    assert res.loc[0, "break_idx"] == 1
    thr_expected = atr(df).iloc[1] * 0.5
    assert res.loc[0, "confirm_threshold"] == pytest.approx(thr_expected)


def test_wick_update_peak():
    idx = pd.date_range("2020", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [0.5, 0.5],
            "high": [1.0, 1.0],
            "low": [0.0, 0.0],
            "close": [0.5, 0.6],
        },
        index=idx,
    )
    levels = pd.DataFrame(
        [{"time": idx[0], "type": "peak", "price": 0.5, "start_idx": 0, "end_idx": 0}]
    )
    cfg = LevelsCfgBreakUpdate(break_mode="atr", atr_mult=0.5)
    res = apply_break_update(df, levels, cfg)
    assert res.loc[0, "state"] == "intact"
    assert res.loc[0, "update_count"] == 1
    assert res.loc[0, "touch_count"] == 1
    assert res.loc[0, "price"] == pytest.approx(1.0)
    assert pd.notna(res.loc[0, "first_touch_time"])


def test_break_body_trough():
    idx = pd.date_range("2020", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [0.5, 0.5],
            "high": [1.0, 0.5],
            "low": [0.0, -0.5],
            "close": [0.5, -0.2],
        },
        index=idx,
    )
    levels = pd.DataFrame(
        [{"time": idx[0], "type": "trough", "price": 0.5, "start_idx": 0, "end_idx": 0}]
    )
    cfg = LevelsCfgBreakUpdate(break_mode="atr", atr_mult=0.5)
    res = apply_break_update(df, levels, cfg)
    assert res.loc[0, "state"] == "broken"
    assert res.loc[0, "break_idx"] == 1


def test_wick_update_trough():
    idx = pd.date_range("2020", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [0.5, 0.5],
            "high": [1.0, 1.0],
            "low": [0.0, 0.0],
            "close": [0.5, 0.4],
        },
        index=idx,
    )
    levels = pd.DataFrame(
        [{"time": idx[0], "type": "trough", "price": 0.5, "start_idx": 0, "end_idx": 0}]
    )
    cfg = LevelsCfgBreakUpdate(break_mode="atr", atr_mult=0.5)
    res = apply_break_update(df, levels, cfg)
    assert res.loc[0, "state"] == "intact"
    assert res.loc[0, "update_count"] == 1
    assert res.loc[0, "touch_count"] == 1
    assert res.loc[0, "price"] == pytest.approx(0.0)
    assert pd.notna(res.loc[0, "first_touch_time"])


def test_update_limit():
    idx = pd.date_range("2020", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [0.5, 0.5],
            "high": [1.0, 2.0],
            "low": [0.0, 0.0],
            "close": [0.5, 0.6],
        },
        index=idx,
    )
    levels = pd.DataFrame(
        [{"time": idx[0], "type": "peak", "price": 0.5, "start_idx": 0, "end_idx": 0}]
    )
    cfg = LevelsCfgBreakUpdate(
        break_mode="atr", atr_mult=0.5, max_update_distance_atr_mult=0.5
    )
    res = apply_break_update(df, levels, cfg)
    assert res.loc[0, "price"] == pytest.approx(0.5)
    assert res.loc[0, "update_count"] == 0
    assert res.loc[0, "touch_count"] == 1


def test_mode_pips_threshold():
    idx = pd.date_range("2020", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.005, 1.005],
            "low": [0.995, 0.995],
            "close": [1.0, 1.002],
        },
        index=idx,
    )
    levels = pd.DataFrame(
        [{"time": idx[0], "type": "peak", "price": 1.0, "start_idx": 0, "end_idx": 0}]
    )
    cfg = LevelsCfgBreakUpdate(break_mode="pips", pips=10, pip_size=0.0001)
    res = apply_break_update(df, levels, cfg)
    assert res.loc[0, "state"] == "broken"
    assert res.loc[0, "break_idx"] == 1
    assert res.loc[0, "confirm_threshold"] == pytest.approx(0.001)


def test_integration_levels_break(tmp_path):
    data_dir = tmp_path / "data"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(data_dir),
    )
    parquet_path = data_dir / "ohlc.parquet"
    levels_dir = tmp_path / "levels"
    analyze_levels_formation(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(levels_dir),
    )
    levels_csv = levels_dir / "levels_formation.csv"
    prop_dir = tmp_path / "prop"
    analyze_levels_prop(
        parquet=str(parquet_path),
        levels_csv=str(levels_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(prop_dir),
    )
    levels_prop_csv = prop_dir / "levels_prop.csv"
    outdir = tmp_path / "break"
    analyze_levels_break(
        parquet=str(parquet_path),
        levels_csv=str(levels_prop_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
    )
    csv_path = outdir / "levels_state.csv"
    assert csv_path.exists()
    state_df = pd.read_csv(csv_path, parse_dates=["time", "break_time", "first_touch_time"])
    required = {
        "state",
        "break_time",
        "break_idx",
        "last_extreme",
        "update_count",
        "touch_count",
        "first_touch_time",
        "confirm_threshold",
        "break_mode",
        "tick_size",
    }
    assert required.issubset(state_df.columns)
    non_null_cols = required - {"break_time", "first_touch_time"}
    assert not state_df[list(non_null_cols)].isna().any().any()
    share_broken = state_df["state"].eq("broken").mean()
    assert 0.0 <= share_broken <= 1.0
