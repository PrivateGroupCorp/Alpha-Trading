from pathlib import Path
import sys
import json

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


def _sample_df():
    idx = pd.date_range("2020", periods=6, freq="H", tz="UTC")
    close = [10.0, 11.0, 10.5, 10.7, 10.6, 10.8]
    open_ = close
    high = [c + 0.2 for c in close]
    low = [c - 0.2 for c in close]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_merge_same_type():
    df = _sample_df()
    levels = pd.DataFrame([
        {"time": df.index[1], "end_idx": 1, "type": "peak", "price": 11.0},
        {"time": df.index[2], "end_idx": 2, "type": "peak", "price": 12.0},
    ])
    res = build_swings(df, levels, SwingsCfg())
    assert len(res) == 1
    assert res.loc[0, "price"] == 12.0
    assert res.loc[0, "merged_count"] == 2


def test_merge_nearby_price():
    df = _sample_df()
    levels = pd.DataFrame([
        {"time": df.index[1], "end_idx": 1, "type": "trough", "price": 10.9},
        {"time": df.index[2], "end_idx": 2, "type": "peak", "price": 10.91},
    ])
    cfg = SwingsCfg(swing_merge_atr_mult=1.0, min_price_delta_atr_mult=0.0, min_gap_bars=1)
    res = build_swings(df, levels, cfg)
    assert len(res) == 1


def test_min_gap_bars():
    df = _sample_df()
    levels = pd.DataFrame([
        {"time": df.index[1], "end_idx": 1, "type": "trough", "price": 10.9},
        {"time": df.index[2], "end_idx": 2, "type": "peak", "price": 11.2},
    ])
    cfg = SwingsCfg(min_gap_bars=3, swing_merge_atr_mult=0.0, min_price_delta_atr_mult=0.0)
    res = build_swings(df, levels, cfg)
    assert len(res) == 1


def test_use_only_strong_swings():
    df = _sample_df()
    levels = pd.DataFrame([
        {"time": df.index[1], "end_idx": 1, "type": "peak", "price": 11.0, "weak_prop": True},
        {"time": df.index[2], "end_idx": 2, "type": "trough", "price": 10.5, "weak_prop": False},
        {"time": df.index[3], "end_idx": 3, "type": "peak", "price": 11.5, "weak_prop": False},
    ])
    res = build_swings(df, levels, SwingsCfg(use_only_strong_swings=True))
    used_ids = [i for ids in res["src_level_ids"] for i in ids]
    assert len(res) == 2
    assert 0 not in used_ids
    assert not res["weak_any"].any()


def test_integration_swings(tmp_path):
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
    struct_dir = tmp_path / "structure"
    analyze_structure_swings(
        parquet=str(parquet_path),
        levels_csv=str(prop_dir / "levels_prop.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(struct_dir),
    )
    swings_path = struct_dir / "swings.csv"
    summary_path = struct_dir / "swings_summary.json"
    assert swings_path.exists()
    assert summary_path.exists()
    swings_df = pd.read_csv(swings_path)
    assert not (swings_df["type"].shift(1) == swings_df["type"]).any()
    if len(swings_df) > 1:
        assert swings_df["leg_from_prev"].iloc[1:].mean() > 0
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_swings"] == len(swings_df)
