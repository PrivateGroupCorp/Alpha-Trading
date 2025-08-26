from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.structure.swings import SwingsCfg, build_swings
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_structure_swings,
)


def _sample_df(n=10):
    idx = pd.date_range("2020", periods=n, freq="H", tz="UTC")
    price = pd.Series(range(n), index=idx, dtype="float64") + 10
    df = pd.DataFrame(
        {
            "open": price + 0.1,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
        }
    )
    return df


def test_merge_same_type():
    df = _sample_df(6)
    levels = pd.DataFrame(
        [
            {"time": df.index[1], "type": "peak", "price": 11.0, "end_idx": 1},
            {"time": df.index[2], "type": "peak", "price": 12.0, "end_idx": 2},
            {"time": df.index[4], "type": "trough", "price": 14.0, "end_idx": 4},
        ]
    )
    cfg = SwingsCfg(atr_window=3)
    swings = build_swings(df, levels, cfg)
    assert len(swings) == 2
    first = swings.iloc[0]
    assert first["type"] == "peak"
    assert first["price"] == 12.0


def test_merge_nearby_price():
    df = _sample_df(5)
    levels = pd.DataFrame(
        [
            {"time": df.index[1], "type": "trough", "price": 11.0, "end_idx": 1},
            {"time": df.index[3], "type": "peak", "price": 11.1, "end_idx": 3},
        ]
    )
    cfg = SwingsCfg(swing_merge_atr_mult=0.3, atr_window=3)
    swings = build_swings(df, levels, cfg)
    assert len(swings) == 1


def test_min_gap_merge():
    df = _sample_df(5)
    levels = pd.DataFrame(
        [
            {"time": df.index[1], "type": "trough", "price": 11.0, "end_idx": 1},
            {"time": df.index[2], "type": "peak", "price": 12.5, "end_idx": 2},
        ]
    )
    cfg = SwingsCfg(swing_merge_atr_mult=0.0, min_gap_bars=2, min_price_delta_atr_mult=0.0, atr_window=3)
    swings = build_swings(df, levels, cfg)
    assert len(swings) == 1


def test_filter_weak_levels():
    df = _sample_df(5)
    levels = pd.DataFrame(
        [
            {"time": df.index[1], "type": "peak", "price": 11.0, "end_idx": 1, "weak_prop": True},
            {"time": df.index[2], "type": "trough", "price": 10.0, "end_idx": 2, "weak_prop": False},
        ]
    )
    cfg = SwingsCfg(atr_window=3)
    swings = build_swings(df, levels, cfg)
    assert len(swings) == 1
    assert swings.iloc[0]["type"] == "trough"


def test_integration_structure_swings(tmp_path):
    data_dir = tmp_path / "data"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(data_dir),
    )
    parquet = data_dir / "ohlc.parquet"
    levels_dir = tmp_path / "levels"
    analyze_levels_formation(
        parquet=str(parquet),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(levels_dir),
    )
    analyze_levels_prop(
        parquet=str(parquet),
        levels_csv=str(levels_dir / "levels_formation.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(levels_dir),
    )
    outdir = tmp_path / "swings"
    analyze_structure_swings(
        parquet=str(parquet),
        levels_csv=str(levels_dir / "levels_prop.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
    )

    swings_csv = outdir / "swings.csv"
    assert swings_csv.exists()
    swings_df = pd.read_csv(swings_csv, parse_dates=["time"])
    assert not (swings_df["type"].shift() == swings_df["type"]).any()
    if len(swings_df) > 1:
        assert swings_df["leg_from_prev"].iloc[1:].mean() > 0
    summary_path = outdir / "swings_summary.json"
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_swings"] == len(swings_df)
    assert summary["n_levels_used"] == int(swings_df["merged_count"].sum())
