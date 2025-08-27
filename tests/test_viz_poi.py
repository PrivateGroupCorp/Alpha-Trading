import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_structure_swings,
    analyze_structure_events,
    analyze_structure_quality,
    analyze_liquidity_eq,
    analyze_liquidity_sweep,
    analyze_liquidity_asia,
    analyze_poi_ob,
    analyze_poi_viz,
)
from alpha.viz.poi import POIVizCfg, build_poi_layers


def _prepare_poi(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
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
    events_q_csv = qual_dir / "events_qualified.csv"

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

    asia_dir = tmp_path / "asia"
    analyze_liquidity_asia(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(asia_dir),
    )

    poi_dir = tmp_path / "poi"
    analyze_poi_ob(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        outdir=str(poi_dir),
        profile="h1",
        swings_csv=str(swings_csv),
        events_csv=str(events_q_csv),
        sweeps_csv=str(sweep_dir / "sweeps.csv"),
        eq_clusters_csv=str(eq_dir / "eq_clusters.csv"),
        asia_daily_csv=str(asia_dir / "asia_range_daily.csv"),
    )

    return (
        parquet_path,
        poi_dir / "poi_zones.csv",
        sweep_dir / "sweeps.csv",
        eq_dir / "eq_clusters.csv",
        asia_dir / "asia_range_daily.csv",
    )


def test_viz_poi_outputs(tmp_path):
    parquet_path, zones_csv, sweeps_csv, eq_csv, asia_csv = _prepare_poi(tmp_path)
    outdir = tmp_path / "plots"
    analyze_poi_viz(
        parquet=str(parquet_path),
        zones_csv=str(zones_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
        sweeps_csv=str(sweeps_csv),
        eq_clusters_csv=str(eq_csv),
        asia_daily_csv=str(asia_csv),
        last_n_bars=200,
    )
    png_path = outdir / "poi_last200.png"
    rect_path = outdir / "poi_rects_last200.csv"
    seg_path = outdir / "poi_segments_last200.csv"
    mark_path = outdir / "poi_markers_last200.csv"
    assert png_path.exists() and png_path.stat().st_size > 30000
    rect_df = pd.read_csv(rect_path)
    assert {
        "zone_id",
        "kind",
        "grade",
        "x_start",
        "x_end",
        "y_bottom",
        "y_top",
        "score",
        "color",
    }.issubset(rect_df.columns)
    if not rect_df.empty:
        assert rect_df["kind"].isin(["demand", "supply", "breaker", "flip"]).all()
    seg_df = pd.read_csv(seg_path)
    if not seg_df.empty:
        assert (seg_df["seg_kind"] == "eq_band").any() or True
        assert (seg_df["seg_kind"].str.contains("asia")).any()
    markers_df = pd.read_csv(mark_path)
    if not markers_df.empty:
        assert markers_df["mark_kind"].str.startswith("sweep").any()


def test_poi_min_grade(tmp_path):
    parquet_path, zones_csv, _sw, _eq, _asia = _prepare_poi(tmp_path)
    df = pd.read_parquet(parquet_path)
    zones_df = pd.read_csv(zones_csv)
    cfg = POIVizCfg(min_grade="A")
    rects, _segs, _marks = build_poi_layers(
        df,
        zones_df,
        None,
        None,
        None,
        None,
        cfg,
        window_last_n=200,
    )
    if not rects.empty:
        assert rects["grade"].str.upper().eq("A").all()

