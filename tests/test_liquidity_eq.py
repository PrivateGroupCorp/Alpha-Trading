from pathlib import Path
import sys

# Ensure project root on path when running via `uv run`
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import json

from alpha.liquidity.eq_clusters import (
    EqCfg,
    detect_eq_touches,
    build_eq_clusters,
    score_clusters,
    summarize_eq,
)
from alpha.app.cli import analyze_liquidity_eq


def _base_df(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    open_ = np.full(n, 1.0)
    close = np.full(n, 1.0)
    high = 1.005 + np.linspace(0, 0.009, n)
    low = 0.995 - np.linspace(0, 0.009, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_single_eqh_cluster():
    df = _base_df()
    df.loc[df.index[5], ["high", "close"]] = [1.0015, 1.0015]
    df.loc[df.index[10], ["high", "close"]] = [1.00155, 1.00155]
    cfg = EqCfg(atr_window=1, eq_atr_tol=0.0, alt_pip_tol=1, min_age_bars=1, merge_gap_atr_mult=0.0)
    touches = detect_eq_touches(df, cfg)
    clusters = build_eq_clusters(df, touches, cfg)
    assert len(clusters) == 1
    cl = clusters.iloc[0]
    assert cl["side"] == "eqh"
    assert cl["touch_count"] == 2
    assert cl["width_atr"] < 0.5


def test_two_clusters_split_by_gap():
    df = _base_df()
    df.loc[df.index[5], ["high", "close"]] = [1.0015, 1.0015]
    df.loc[df.index[10], ["high", "close"]] = [1.00155, 1.00155]
    df.loc[df.index[20], ["high", "close"]] = [1.0030, 1.0030]
    df.loc[df.index[25], ["high", "close"]] = [1.00305, 1.00305]
    cfg = EqCfg(atr_window=1, eq_atr_tol=0.0, alt_pip_tol=1, min_age_bars=1, merge_gap_atr_mult=0.0)
    touches = detect_eq_touches(df, cfg)
    clusters = build_eq_clusters(df, touches, cfg)
    assert len(clusters) == 2


def test_cleanliness_less_than_one():
    df = _base_df()
    df.loc[df.index[5], ["high", "close"]] = [1.0015, 1.0015]
    df.loc[df.index[10], ["high", "close"]] = [1.00155, 1.00155]
    df.loc[df.index[7], "close"] = 1.002  # close beyond cluster
    cfg = EqCfg(atr_window=1, eq_atr_tol=0.0, alt_pip_tol=1, min_age_bars=1, merge_gap_atr_mult=0.0)
    touches = detect_eq_touches(df, cfg)
    clusters = build_eq_clusters(df, touches, cfg)
    cl = clusters.iloc[0]
    assert cl["clean_bodies_ratio"] < 1.0


def test_scoring_increases_with_touches():
    df = _base_df()
    df.loc[df.index[5], ["high", "close"]] = [1.0015, 1.0015]
    df.loc[df.index[10], ["high", "close"]] = [1.00155, 1.00155]
    df.loc[df.index[15], ["high", "close"]] = [1.00152, 1.00152]
    df.loc[df.index[25], ["high", "close"]] = [1.0030, 1.0030]
    df.loc[df.index[30], ["high", "close"]] = [1.00305, 1.00305]
    cfg = EqCfg(atr_window=1, eq_atr_tol=0.0, alt_pip_tol=1, min_age_bars=1, merge_gap_atr_mult=0.0)
    touches = detect_eq_touches(df, cfg)
    clusters = build_eq_clusters(df, touches, cfg)
    clusters = score_clusters(clusters, cfg)
    scores = clusters.sort_values("touch_count")["score"].to_list()
    assert scores[0] < scores[-1]


def test_cli_outputs(tmp_path):
    df = _base_df(60)
    df.loc[df.index[10], ["high", "close"]] = [1.0015, 1.0015]
    df.loc[df.index[20], ["high", "close"]] = [1.00155, 1.00155]
    df.loc[df.index[30], ["low", "close"]] = [0.9985, 0.9985]
    df.loc[df.index[40], ["low", "close"]] = [0.99845, 0.99845]
    parquet_path = tmp_path / "ohlc.parquet"
    df.to_parquet(parquet_path)

    outdir = tmp_path / "out"
    analyze_liquidity_eq(str(parquet_path), "TEST", "M15", str(outdir), profile="h1")

    touches_csv = outdir / "eq_touches.csv"
    clusters_csv = outdir / "eq_clusters.csv"
    summary_json = outdir / "eq_clusters_summary.json"
    segments_csv = outdir / "eq_segments.csv"

    assert touches_csv.exists() and clusters_csv.exists() and summary_json.exists()
    touches_df = pd.read_csv(touches_csv)
    clusters_df = pd.read_csv(clusters_csv, parse_dates=["first_time", "last_time"])
    with open(summary_json, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    assert len(touches_df) > 0
    assert len(clusters_df) > 0
    assert set(clusters_df["side"]) == {"eqh", "eql"}
    assert 0.0 <= clusters_df["score"].median() <= 1.0
    assert summary["n_touches"] == len(touches_df)
    assert summary["n_clusters_valid"] == len(clusters_df)
    assert segments_csv.exists()
