from pathlib import Path
import sys

# Ensure project root on path when running via `uv run`
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_levels_break,
    analyze_levels_viz,
)


def _prepare_levels(tmp_path: Path) -> tuple[Path, Path]:
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
    break_dir = tmp_path / "break"
    analyze_levels_break(
        parquet=str(parquet_path),
        levels_csv=str(prop_dir / "levels_prop.csv"),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(break_dir),
    )
    return parquet_path, break_dir / "levels_state.csv"


def test_viz_levels_outputs(tmp_path):
    parquet_path, levels_state = _prepare_levels(tmp_path)
    outdir = tmp_path / "plots"
    analyze_levels_viz(
        parquet=str(parquet_path),
        levels_csv=str(levels_state),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
        last_n_bars=50,
    )
    png_path = outdir / "levels_last50.png"
    seg_path = outdir / "levels_segments_last50.csv"
    mark_path = outdir / "levels_markers_last50.csv"
    assert png_path.exists() and png_path.stat().st_size > 1000
    assert seg_path.exists()
    seg_df = pd.read_csv(seg_path)
    expected_cols = {
        "type",
        "state",
        "y",
        "t_start",
        "t_end",
        "price_at_plot",
        "break_idx",
        "first_touch_time",
        "update_count",
    }
    assert expected_cols.issubset(seg_df.columns)
    assert mark_path.exists()
    markers_df = pd.read_csv(mark_path)
    assert {"kind", "time", "y", "level_type", "state"}.issubset(markers_df.columns)


def test_viz_levels_no_levels(tmp_path):
    parquet_path, _ = _prepare_levels(tmp_path)
    empty_levels = tmp_path / "empty.csv"
    pd.DataFrame(columns=["time", "type", "price", "start_idx", "end_idx"]).to_csv(
        empty_levels, index=False
    )
    outdir = tmp_path / "plots"
    analyze_levels_viz(
        parquet=str(parquet_path),
        levels_csv=str(empty_levels),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
        last_n_bars=50,
    )
    png_path = outdir / "levels_last50.png"
    assert png_path.exists() and png_path.stat().st_size > 1000
    seg_path = outdir / "levels_segments_last50.csv"
    seg_df = pd.read_csv(seg_path)
    assert seg_df.empty
