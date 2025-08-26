from pathlib import Path
import sys

import numpy as np
import pandas as pd
import json

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.liquidity.sweep import SweepCfg, detect_sweeps, summarize_sweeps
from alpha.app.cli import analyze_liquidity_sweep


def _make_df_basic():
    idx = pd.date_range("2023-01-01", periods=10, freq="H", tz="UTC")
    base = 1.0
    open_ = np.full(len(idx), base)
    close = np.full(len(idx), base)
    high = np.full(len(idx), base + 0.03)
    low = np.full(len(idx), base - 0.03)
    # penetration at bar 3 with body close below edge
    close[3] = base - 0.06
    low[3] = base - 0.06
    # reclaim next bar
    close[4] = base + 0.01
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _eq_clusters_basic(df):
    return pd.DataFrame(
        {
            "cluster_id": [1],
            "side": ["eql"],
            "price_center": [float(df["low"].iloc[0])],
            "width_atr": [0.02],
            "score": [0.8],
            "first_time": [df.index[0]],
            "last_time": [df.index[2]],
        }
    )


def test_detect_sweep_basic():
    df = _make_df_basic()
    clusters = _eq_clusters_basic(df)
    cfg = SweepCfg(atr_window=1, pip_size=0.01)
    sweeps = detect_sweeps(df, clusters, pd.DataFrame(), None, cfg)
    assert len(sweeps) == 1
    row = sweeps.iloc[0]
    assert bool(row["reclaim_ok"]) is True
    assert row["duration_bars"] == 1
    assert row["quality_grade"] in {"A", "B", "C"}


def test_penetration_too_large():
    df = _make_df_basic()
    df.loc[df.index[3], ["low", "close"]] = 0.5
    clusters = _eq_clusters_basic(df)
    cfg = SweepCfg(atr_window=3, pip_size=0.01)
    sweeps = detect_sweeps(df, clusters, pd.DataFrame(), None, cfg)
    assert sweeps.empty


def test_confirm_wick_mode():
    df = _make_df_basic()
    # Wick-only penetration
    df.loc[df.index[3], "close"] = 1.0
    clusters = _eq_clusters_basic(df)
    cfg = SweepCfg(confirm_mode="wick", atr_window=1, pip_size=0.01)
    sweeps = detect_sweeps(df, clusters, pd.DataFrame(), None, cfg)
    assert len(sweeps) == 1


def test_reclaim_close_n_bars_fail():
    df = _make_df_basic()
    # Delay reclaim beyond n bars
    df.loc[df.index[4], "close"] = 0.95
    df.loc[df.index[5], "close"] = 0.95
    df.loc[df.index[6], "close"] = 1.01
    clusters = _eq_clusters_basic(df)
    cfg = SweepCfg(reclaim_mode="close_back_n_bars", reclaim_n_bars=2, atr_window=1, pip_size=0.01)
    sweeps = detect_sweeps(df, clusters, pd.DataFrame(), None, cfg)
    assert len(sweeps) == 1
    assert bool(sweeps.iloc[0]["reclaim_ok"]) is False


def test_cli_sweep_outputs(tmp_path):
    df = _make_df_basic()
    parquet_path = tmp_path / "ohlc.parquet"
    df.to_parquet(parquet_path)
    clusters = _eq_clusters_basic(df)
    clusters_csv = tmp_path / "eq_clusters.csv"
    clusters.to_csv(clusters_csv, index=False)

    outdir = tmp_path / "out"
    analyze_liquidity_sweep(
        parquet=str(parquet_path),
        symbol="TEST",
        tf="H1",
        outdir=str(outdir),
        profile="h1",
        eq_clusters_csv=str(clusters_csv),
    )

    sweeps_csv = outdir / "sweeps.csv"
    summary_json = outdir / "sweeps_summary.json"
    assert sweeps_csv.exists() and summary_json.exists()
    sweeps_df = pd.read_csv(sweeps_csv)
    with open(summary_json, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert len(sweeps_df) == 1
    assert summary["n_sweeps"] == 1
