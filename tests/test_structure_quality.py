from pathlib import Path
import sys
import json

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.structure.quality import QualityCfg, qualify_events
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_structure_swings,
    analyze_structure_events,
    analyze_structure_quality,
)


def _sample_df():
    idx = pd.date_range("2020", periods=10, freq="H", tz="UTC")
    close = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 10.85, 11.2]
    open_ = close
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
    low[8] = 10.78  # deeper wick for retest
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _sample_events(idx):
    return pd.DataFrame(
        [
            {
                "event_id": 0,
                "time": idx[1],
                "idx": 1,
                "event": "BOS",
                "direction": "up",
                "ref_price": 10.2,
                "break_margin_norm": 0.10,
            },
            {
                "event_id": 1,
                "time": idx[5],
                "idx": 5,
                "event": "BOS",
                "direction": "up",
                "ref_price": 10.8,
                "break_margin_norm": 0.50,
            },
        ]
    )


def test_quality_basic_features():
    df = _sample_df()
    events = _sample_events(df.index)
    cfg = QualityCfg()
    q = qualify_events(df, events, cfg)
    assert not q.loc[0, "is_valid"]  # low break margin
    assert q.loc[1, "is_valid"]  # passes all gates
    assert q.loc[1, "has_retest"]
    assert q.loc[1, "has_sweep"]
    assert q.loc[1, "quality_score"] > q.loc[0, "quality_score"]


def test_integration_quality(tmp_path):
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
    qual_dir = tmp_path / "quality"
    analyze_structure_quality(
        parquet=str(parquet_path),
        events_csv=str(events_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(qual_dir),
    )
    qual_csv = qual_dir / "events_qualified.csv"
    summary_path = qual_dir / "events_quality_summary.json"
    assert qual_csv.exists()
    assert summary_path.exists()
    qdf = pd.read_csv(qual_csv)
    for col in [
        "is_valid",
        "break_margin_norm_pass",
        "ft_bars",
        "ft_pass",
        "ft_avg_dist_atr",
        "has_retest",
        "has_sweep",
        "quality_score",
        "quality_grade",
    ]:
        assert col in qdf.columns
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_events_in"] == len(qdf)
    assert 0.0 <= summary["share_valid"] <= 1.0
