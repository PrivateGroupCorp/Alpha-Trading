from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.poi.ob import PoiCfg, build_poi_zones
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_structure_swings,
    analyze_structure_events,
    analyze_structure_quality,
    analyze_liquidity_eq,
    analyze_liquidity_sweep,
    analyze_poi_ob,
)


def _basic_df():
    idx = pd.date_range("2023", periods=6, freq="H", tz="UTC")
    open_ = [10.0, 10.2, 10.4, 10.3, 10.5, 10.6]
    high = [10.2, 10.4, 10.46, 10.6, 10.9, 10.7]
    low = [9.8, 10.0, 10.1, 10.2, 10.4, 10.5]
    close = [10.0, 10.3, 10.2, 10.55, 10.85, 10.6]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_demand_ob_basic():
    df = _basic_df()
    events = pd.DataFrame([
        {
            "event_id": 0,
            "idx": 4,
            "event": "BOS",
            "direction": "up",
            "break_margin_norm": 0.5,
            "leg_from_origin": 0.6,
            "atr_at_break": 0.3,
            "is_valid": True,
        }
    ])
    cfg = PoiCfg(ob_lookback_bars=3, ob_zone_padding_atr_mult=0.0, atr_window=1,
                 pip_size=0.1, min_width_atr=0.0, max_width_atr=10.0)
    zones = build_poi_zones(df, pd.DataFrame(), events, None, None, None, None, cfg)
    assert len(zones) == 1
    z = zones.iloc[0]
    assert z.kind == "demand"
    assert z.anchor_idx == 2
    assert z.price_top == pytest.approx(10.4)
    assert z.price_bottom == pytest.approx(10.1)


def test_supply_ob_basic():
    df = _basic_df()
    events = pd.DataFrame([
        {
            "event_id": 0,
            "idx": 4,
            "event": "BOS",
            "direction": "down",
            "break_margin_norm": 0.5,
            "leg_from_origin": 0.6,
            "atr_at_break": 0.3,
            "is_valid": True,
        }
    ])
    cfg = PoiCfg(ob_lookback_bars=3, ob_zone_padding_atr_mult=0.0, atr_window=1,
                 pip_size=0.1, min_width_atr=0.0, max_width_atr=10.0)
    zones = build_poi_zones(df, pd.DataFrame(), events, None, None, None, None, cfg)
    assert len(zones) == 1
    z = zones.iloc[0]
    assert z.kind == "supply"
    assert z.price_top == pytest.approx(10.6)
    assert z.price_bottom == pytest.approx(10.3)


def test_fvg_detection():
    idx = pd.date_range("2023", periods=4, freq="H", tz="UTC")
    df = pd.DataFrame({
        "open": [10.0, 10.2, 10.35, 10.4],
        "high": [10.1, 10.25, 10.4, 10.5],
        "low": [9.9, 10.1, 10.3, 10.35],
        "close": [10.0, 10.15, 10.38, 10.45],
    }, index=idx)
    events = pd.DataFrame([{ "event_id": 0, "idx": 3, "event": "BOS", "direction": "up",
                             "break_margin_norm": 0.5, "leg_from_origin": 0.5,
                             "atr_at_break": 0.2, "is_valid": True }])
    cfg = PoiCfg(ob_lookback_bars=3, ob_zone_padding_atr_mult=0.0, atr_window=1,
                 pip_size=0.1, min_width_atr=0.0, max_width_atr=10.0)
    zones = build_poi_zones(df, pd.DataFrame(), events, None, None, None, None, cfg)
    assert not zones.empty and bool(zones.iloc[0].fvg_present)


def test_near_sweep_scoring():
    df = _basic_df()
    events = pd.DataFrame([
        {
            "event_id": 0,
            "idx": 4,
            "event": "BOS",
            "direction": "up",
            "break_margin_norm": 0.5,
            "leg_from_origin": 0.6,
            "atr_at_break": 0.3,
            "is_valid": True,
        }
    ])
    sweeps = pd.DataFrame([{ "sweep_id": 1, "pen_idx": 4, "quality_grade": "A" }])
    cfg = PoiCfg(ob_lookback_bars=3, ob_zone_padding_atr_mult=0.0, atr_window=1,
                 pip_size=0.1, min_width_atr=0.0, max_width_atr=10.0)
    z1 = build_poi_zones(df, pd.DataFrame(), events, None, None, None, None, cfg)
    z2 = build_poi_zones(df, pd.DataFrame(), events, sweeps, None, None, None, cfg)
    assert z2.iloc[0].score_total > z1.iloc[0].score_total


def test_merge_overlap():
    idx = pd.date_range("2023", periods=6, freq="H", tz="UTC")
    df = pd.DataFrame({
        "open": [10.0, 10.2, 10.5, 10.7, 10.8, 11.0],
        "high": [10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
        "low": [9.8, 10.0, 10.3, 10.5, 10.6, 10.8],
        "close": [10.1, 10.3, 10.4, 10.6, 10.7, 11.1],
    }, index=idx)
    events = pd.DataFrame([
        {"event_id": 0, "idx": 4, "event": "BOS", "direction": "up",
         "break_margin_norm": 0.5, "leg_from_origin": 0.6,
         "atr_at_break": 0.3, "is_valid": True},
        {"event_id": 1, "idx": 5, "event": "BOS", "direction": "up",
         "break_margin_norm": 0.5, "leg_from_origin": 0.6,
         "atr_at_break": 0.3, "is_valid": True},
    ])
    cfg = PoiCfg(ob_lookback_bars=3, ob_zone_padding_atr_mult=0.0, atr_window=1,
                 pip_size=0.1, min_width_atr=0.0, max_width_atr=10.0)
    zones = build_poi_zones(df, pd.DataFrame(), events, None, None, None, None, cfg)
    assert len(zones) == 1
    z = zones.iloc[0]
    assert z.price_bottom <= 10.5 and z.price_top >= 10.8


def test_cli_integration_poi(tmp_path):
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
    prop_dir = tmp_path / "prop"
    analyze_levels_prop(
        parquet=str(parquet_path),
        levels_csv=str(levels_dir / "levels_formation.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(prop_dir),
    )
    struct_dir = tmp_path / "struct"
    analyze_structure_swings(
        parquet=str(parquet_path),
        levels_csv=str(prop_dir / "levels_prop.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(struct_dir),
    )
    swings_csv = struct_dir / "swings.csv"
    events_dir = tmp_path / "events"
    analyze_structure_events(
        parquet=str(parquet_path),
        swings_csv=str(swings_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(events_dir),
    )
    qual_dir = tmp_path / "quality"
    analyze_structure_quality(
        parquet=str(parquet_path),
        events_csv=str(events_dir / "events.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(qual_dir),
    )
    eq_dir = tmp_path / "eq"
    analyze_liquidity_eq(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(eq_dir),
    )
    sweep_dir = tmp_path / "sweep"
    analyze_liquidity_sweep(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        outdir=str(sweep_dir),
        profile="h1",
        eq_clusters_csv=str(eq_dir / "eq_clusters.csv"),
    )
    poi_dir = tmp_path / "poi"
    analyze_poi_ob(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        outdir=str(poi_dir),
        profile="h1",
        swings_csv=str(swings_csv),
        events_csv=str(qual_dir / "events_qualified.csv"),
        sweeps_csv=str(sweep_dir / "sweeps.csv"),
        eq_clusters_csv=str(eq_dir / "eq_clusters.csv"),
    )
    zones_csv = poi_dir / "poi_zones.csv"
    summary_json = poi_dir / "poi_summary.json"
    assert zones_csv.exists() and summary_json.exists()
    zones_df = pd.read_csv(zones_csv)
    assert "score_total" in zones_df.columns
    if not zones_df.empty:
        assert zones_df["score_total"].between(0, 1).all()
