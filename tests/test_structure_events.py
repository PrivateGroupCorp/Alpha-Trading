from pathlib import Path
import sys
import json

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.structure.events import EventsCfg, detect_bos_choch
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_structure_swings,
    analyze_structure_events,
)


def _sample_df():
    idx = pd.date_range("2020", periods=7, freq="H", tz="UTC")
    close = [10.0, 11.0, 9.0, 12.0, 13.0, 10.0, 8.0]
    open_ = close
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _sample_swings(idx):
    return pd.DataFrame(
        [
            {"swing_id": 0, "time": idx[0], "idx": 0, "type": "trough", "price": 10.0},
            {"swing_id": 1, "time": idx[1], "idx": 1, "type": "peak", "price": 11.0},
            {"swing_id": 2, "time": idx[2], "idx": 2, "type": "trough", "price": 9.0},
            {"swing_id": 3, "time": idx[4], "idx": 4, "type": "peak", "price": 13.0},
            {"swing_id": 4, "time": idx[5], "idx": 5, "type": "trough", "price": 10.0},
        ]
    )


def test_detect_bos_choch_basic():
    df = _sample_df()
    swings = _sample_swings(df.index)
    cfg = EventsCfg(bos_break_mult_atr=0.05, bos_leg_min_atr_mult=0.05, atr_window=1, event_cooldown_bars=1)
    events = detect_bos_choch(df, swings, cfg)
    assert list(events["event"]) == ["BOS", "CHoCH"]
    assert list(events["direction"]) == ["up", "down"]
    assert (events["break_margin_norm"] > 0).all()
    assert events.loc[0, "bars_since_ref"] == 2
    assert events.loc[0, "bars_since_origin"] == 1


def test_event_cooldown():
    df = _sample_df()
    swings = _sample_swings(df.index)
    cfg = EventsCfg(bos_break_mult_atr=0.05, bos_leg_min_atr_mult=0.05, atr_window=1, event_cooldown_bars=5)
    events = detect_bos_choch(df, swings, cfg)
    assert len(events) == 1


def test_integration_events(tmp_path):
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
    events_dir = tmp_path / "events"
    analyze_structure_events(
        parquet=str(parquet_path),
        swings_csv=str(swings_path),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(events_dir),
    )
    events_csv = events_dir / "events.csv"
    summary_path = events_dir / "events_summary.json"
    assert events_csv.exists()
    assert summary_path.exists()
    events_df = pd.read_csv(events_csv)
    assert events_df["event"].isin(["BOS", "CHoCH"]).all()
    assert not events_df[["ref_price", "origin_price", "confirm_threshold"]].isna().any().any()
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_events"] == len(events_df)
    assert 0.0 <= summary["median_break_margin_norm"] <= 5.0
