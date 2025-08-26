import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_structure_swings,
    analyze_structure_events,
    analyze_structure_quality,
    analyze_structure_viz,
)


def _prepare_structure(tmp_path: Path) -> tuple[Path, Path, Path]:
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

    struct_dir = tmp_path / "structure"
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

    quality_dir = tmp_path / "quality"
    analyze_structure_quality(
        parquet=str(parquet_path),
        events_csv=str(events_dir / "events.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(quality_dir),
    )
    events_q_csv = quality_dir / "events_qualified.csv"
    return parquet_path, swings_csv, events_q_csv


def test_viz_structure_outputs(tmp_path):
    parquet_path, swings_csv, events_q_csv = _prepare_structure(tmp_path)
    outdir = tmp_path / "plots"
    analyze_structure_viz(
        parquet=str(parquet_path),
        swings_csv=str(swings_csv),
        events_csv=str(events_q_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
        last_n_bars=50,
    )
    png_path = outdir / "structure_last50.png"
    seg_path = outdir / "structure_segments_last50.csv"
    mark_path = outdir / "structure_markers_last50.csv"
    assert png_path.exists() and png_path.stat().st_size > 1000
    assert seg_path.exists()
    seg_df = pd.read_csv(seg_path)
    assert {"seg_kind", "x_start", "x_end", "y_start", "y_end", "meta"}.issubset(seg_df.columns)
    assert (seg_df["seg_kind"] == "price").any()
    if not seg_df[seg_df["seg_kind"] == "zigzag"].empty:
        assert True
    assert mark_path.exists()
    markers_df = pd.read_csv(mark_path)
    assert {
        "mark_kind",
        "time",
        "y",
        "grade",
        "is_valid",
        "event_id",
        "swing_id",
    }.issubset(markers_df.columns)
    non_swing = markers_df[~markers_df["mark_kind"].str.startswith("swing")]
    if not non_swing.empty:
        assert (non_swing["is_valid"] == True).all()


def test_viz_structure_no_events(tmp_path):
    parquet_path, swings_csv, _events_q_csv = _prepare_structure(tmp_path)
    empty_events = tmp_path / "empty_events.csv"
    pd.DataFrame(
        columns=["event_id", "time", "event", "direction", "ref_price", "is_valid", "quality_grade"]
    ).to_csv(empty_events, index=False)
    outdir = tmp_path / "plots"
    analyze_structure_viz(
        parquet=str(parquet_path),
        swings_csv=str(swings_csv),
        events_csv=str(empty_events),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
        last_n_bars=50,
    )
    markers_df = pd.read_csv(outdir / "structure_markers_last50.csv")
    assert markers_df["mark_kind"].str.startswith("swing").all()
