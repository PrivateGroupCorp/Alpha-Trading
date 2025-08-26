from pathlib import Path
import sys

# Ensure project root on path when running via `uv run`
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from alpha.core.indicators import atr, atr_ewm, atr_wilder, is_doji_series, true_range
from alpha.app.cli import analyze_levels_data, indicators


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [10.0, 12.0, 18.0, 15.0],
            "high": [15.0, 16.0, 20.0, 18.0],
            "low": [9.0, 11.0, 17.0, 14.0],
            "close": [12.0, 15.0, 18.0, 16.0],
        },
        index=idx,
    )
    return df


def test_true_range_and_doji_basic():
    df = _sample_df()
    tr = true_range(df)
    expected = pd.Series([6.0, 5.0, 5.0, 4.0], index=df.index, name="tr")
    pd.testing.assert_series_equal(tr, expected)

    # Doji detection
    df_doji = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 2.0, 1.0],
            "low": [1.0, 0.0, 1.0],
            "close": [1.0, 1.5, 1.0],
        },
        index=pd.date_range("2020", periods=3, freq="D", tz="UTC"),
    )
    is_doji = is_doji_series(df_doji, body_ratio=0.2)
    expected_doji = pd.Series([True, False, True], index=df_doji.index, name="is_doji")
    pd.testing.assert_series_equal(is_doji, expected_doji)


def test_atr_methods_unit():
    df = _sample_df()
    atr_e = atr_ewm(df, window=3)
    atr_w = atr_wilder(df, window=3)
    assert atr_e.notna().all()
    assert atr_e.gt(0).all()
    assert atr_w.isna().sum() == 2
    assert atr_w.dropna().gt(0).all()


def test_true_range_edge_cases():
    idx = pd.date_range("2021", periods=2, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [1.0, 5.0],
            "low": [1.0, 4.0],
            "close": [1.0, 4.0],
        },
        index=idx,
    )
    tr = true_range(df)
    assert tr.iloc[0] == 0.0
    assert tr.iloc[1] == 4.0


def test_indicators_integration(tmp_path):
    outdir = tmp_path / "data"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(outdir),
    )
    df = pd.read_parquet(outdir / "ohlc.parquet")
    # replicate data to have enough rows for ATR window
    df = pd.concat([df] * 3)
    df.index = pd.date_range(df.index[0], periods=len(df), freq="h", tz="UTC")
    big_path = outdir / "ohlc_big.parquet"
    df.to_parquet(big_path)

    atr_e = atr(df, 14, method="ewm")
    atr_w = atr(df, 14, method="wilder")
    assert atr_e.notna().all()
    assert atr_e.gt(0).all()
    assert atr_w.isna().sum() == 13
    assert atr_w.dropna().gt(0).all()
    tail = int(len(df) * 0.3)
    assert atr_e.iloc[-tail:].corr(atr_w.iloc[-tail:]) > 0.98

    feat_dir = tmp_path / "feat"
    indicators(
        parquet=str(big_path),
        symbol="EURUSD",
        tf="H1",
        atr_window=14,
        atr_method="ewm",
        doji_body_ratio=0.2,
        outdir=str(feat_dir),
    )
    path = feat_dir / "indicators.parquet"
    assert path.exists()
    res = pd.read_parquet(path)
    assert {"tr", "atr_ewm_14", "is_doji"}.issubset(res.columns)
