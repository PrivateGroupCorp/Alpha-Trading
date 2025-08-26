from pathlib import Path
import sys

# Ensure project root on path when running via `uv run`
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pytest
import json

from alpha.liquidity.asia import AsiaCfg, asia_range_daily, summarize_asia_ranges
from alpha.app.cli import analyze_liquidity_asia


def _make_df_break_up():
    idx = pd.date_range("2023-01-01", periods=60, freq="15min", tz="UTC")
    base = 1.0
    open_ = np.full(len(idx), base)
    close = np.full(len(idx), base)
    high = np.full(len(idx), base + 0.02)
    low = np.full(len(idx), base - 0.02)
    high[36] = base + 0.03
    close[36] = base + 0.03
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _make_df_touch_only():
    idx = pd.date_range("2023-01-01", periods=60, freq="15min", tz="UTC")
    base = 1.0
    open_ = np.full(len(idx), base)
    close = np.full(len(idx), base)
    high = np.full(len(idx), base + 0.02)
    low = np.full(len(idx), base - 0.02)
    # touch but no break at 09:00
    close[36] = base + 0.02
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _make_df_no_interaction():
    idx = pd.date_range("2023-01-01", periods=60, freq="15min", tz="UTC")
    base = 1.0
    open_ = np.full(len(idx), base)
    close = np.full(len(idx), base)
    high = np.full(len(idx), base + 0.019)
    low = np.full(len(idx), base - 0.019)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_break_up_range():
    df = _make_df_break_up()
    cfg = AsiaCfg(atr_window=1, pip_size=0.01)
    daily = asia_range_daily(df, cfg)
    row = daily.iloc[0]
    assert row["label"] == "BreakUp"
    assert row["post_first_break_dir"] == "up"
    assert row["width_pips"] == pytest.approx(4.0)
    assert row["width_atr_norm"] == pytest.approx(1.0)


def test_touch_no_break():
    df = _make_df_touch_only()
    cfg = AsiaCfg(atr_window=1, pip_size=0.01)
    row = asia_range_daily(df, cfg).iloc[0]
    assert row["label"] == "BothTouch_NoBreak"
    assert row["post_first_touch_dir"] == "up"
    assert row["post_first_break_dir"] == "none"


def test_no_interaction():
    df = _make_df_no_interaction()
    cfg = AsiaCfg(atr_window=1, pip_size=0.01)
    row = asia_range_daily(df, cfg).iloc[0]
    assert row["label"] == "NoInteraction"
    assert row["post_first_touch_dir"] == "none"
    assert row["post_first_break_dir"] == "none"


def test_cli_outputs(tmp_path):
    df1 = _make_df_break_up()
    df2 = _make_df_no_interaction()
    df2.index = df2.index + pd.Timedelta(days=1)
    df = pd.concat([df1, df2])
    parquet_path = tmp_path / "ohlc.parquet"
    df.to_parquet(parquet_path)

    outdir = tmp_path / "out"
    analyze_liquidity_asia(str(parquet_path), "TEST", "M15", "h1", str(outdir))

    daily_csv = outdir / "asia_range_daily.csv"
    summary_json = outdir / "asia_range_summary.json"
    assert daily_csv.exists() and summary_json.exists()

    daily_df = pd.read_csv(daily_csv)
    with open(summary_json, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert len(daily_df) == 2
    assert summary["n_days"] == 2
