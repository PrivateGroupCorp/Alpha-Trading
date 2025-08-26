from pathlib import Path
import sys
import json

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.eval.levels_metrics import compute_levels_metrics, score_levels
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
    analyze_levels_break,
    analyze_levels_metrics,
)


def test_compute_metrics_basic():
    idx = pd.date_range("2020", periods=5, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 1, 1],
            "high": [1.2, 1.3, 1.4, 1.5, 1.6],
            "low": [0.8, 0.7, 0.6, 0.5, 0.4],
            "close": [1.0, 1.1, 1.25, 0.9, 1.0],
        },
        index=idx,
    )
    levels_state = pd.DataFrame(
        [
            {
                "id": 1,
                "time": idx[0],
                "type": "peak",
                "state": "broken",
                "end_idx": 0,
                "break_idx": 2,
                "price": 1.0,
                "confirm_threshold": 0.1,
                "update_count": 0,
                "first_touch_time": pd.NaT,
                "weak_prop": False,
            },
            {
                "id": 2,
                "time": idx[1],
                "type": "trough",
                "state": "intact",
                "end_idx": 1,
                "break_idx": -1,
                "price": 0.8,
                "confirm_threshold": 0.1,
                "update_count": 1,
                "first_touch_time": idx[3],
                "weak_prop": True,
            },
        ]
    )

    metrics_df, per_level_df = compute_levels_metrics(df, levels_state)
    m = metrics_df.iloc[0]
    assert m["n_levels"] == 2
    assert m["n_broken"] == 1
    assert m["share_broken"] == 0.5
    assert m["weak_prop_share"] == 0.5
    assert m["ttb_median"] == 2
    assert m["tft_median"] == 2
    assert m["update_mean"] == 0.5
    assert m["break_strength_median"] == pytest.approx(1.5)
    assert not per_level_df.empty

    eval_cfg = {
        "levels_scoring": {
            "weights": {
                "density": 0.30,
                "share_broken": 0.30,
                "weak_prop": 0.15,
                "ttb": 0.15,
                "balance": 0.10,
            },
            "targets": {
                "H1": {
                    "density": {"good": [0, 500], "soft": [0, 500]},
                    "share_broken": {"good": [0.4, 0.6], "soft": [0.3, 0.7]},
                    "weak_prop_share": {"good": [0.0, 0.6], "soft": [0.0, 0.8]},
                    "ttb_median": {"good": [1, 3], "soft": [0, 4]},
                    "balance_peak_trough": {"good": [0.5, 2.0], "soft": [0.5, 2.0]},
                }
            },
        }
    }
    scores = score_levels(m, tf="H1", eval_cfg=eval_cfg)
    assert scores["score_total"] == pytest.approx(1.0)


def test_cli_levels_metrics(tmp_path):
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
    break_dir = tmp_path / "break"
    analyze_levels_break(
        parquet=str(parquet_path),
        levels_csv=str(levels_prop_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(break_dir),
    )
    levels_state_csv = break_dir / "levels_state.csv"
    outdir = tmp_path / "metrics"
    analyze_levels_metrics(
        parquet=str(parquet_path),
        levels_state_csv=str(levels_state_csv),
        symbol="EURUSD",
        tf="H1",
        eval_profile="H1",
        outdir=str(outdir),
    )
    csv_path = outdir / "levels_metrics.csv"
    json_path = outdir / "levels_metrics.json"
    assert csv_path.exists() and json_path.exists()
    metrics = pd.read_csv(csv_path)
    assert {"n_levels", "share_broken", "score_total"}.issubset(metrics.columns)
    with json_path.open("r", encoding="utf-8") as fh:
        j = json.load(fh)
    assert "score_total" in j and 0.0 <= j["score_total"] <= 1.0


def test_metrics_fallback_no_break(tmp_path):
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
    outdir = tmp_path / "metrics"
    analyze_levels_metrics(
        parquet=str(parquet_path),
        levels_state_csv=str(levels_prop_csv),
        symbol="EURUSD",
        tf="H1",
        eval_profile="H1",
        outdir=str(outdir),
    )
    metrics = pd.read_csv(outdir / "levels_metrics.csv")
    assert metrics.loc[0, "n_levels"] >= 0
    assert pd.isna(metrics.loc[0, "ttb_median"])
