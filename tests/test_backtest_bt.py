from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha.backtest.bt_runner import run_backtest_bt, BrokerCfg


def _sample_data(tmp_path):
    times = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": times,
            "open": [1.0000, 1.0005, 1.0015, 1.0026],
            "high": [1.0006, 1.0016, 1.0027, 1.0038],
            "low": [0.9994, 1.0004, 1.0013, 1.0024],
            "close": [1.0005, 1.0015, 1.0026, 1.0037],
            "volume": [1, 1, 1, 1],
        }
    )
    df = df.set_index("time")
    m1 = tmp_path / "ohlc.parquet"
    df.to_parquet(m1)
    zones = pd.DataFrame(
        {
            "zone_id": [1],
            "kind": ["demand"],
            "price_top": [1.0006],
            "price_bottom": [0.9994],
            "grade": ["A"],
        }
    )
    zcsv = tmp_path / "zones.csv"
    zones.to_csv(zcsv, index=False)
    return m1, zcsv


def test_basic_backtest_bt(tmp_path):
    m1, zones = _sample_data(tmp_path)
    outdir = tmp_path / "out"
    run_backtest_bt(
        str(m1),
        str(zones),
        symbol="EURUSD",
        htf="H1",
        outdir=str(outdir),
        broker_cfg_override=BrokerCfg(commission_per_million=0),
    )
    trades = pd.read_csv(outdir / "trades.csv")
    assert len(trades) >= 1
    assert trades["trade_id"].nunique() >= 1


def test_commission_effect(tmp_path):
    m1, zones = _sample_data(tmp_path)
    out0 = tmp_path / "o0"
    run_backtest_bt(
        str(m1),
        str(zones),
        symbol="EURUSD",
        htf="H1",
        outdir=str(out0),
        broker_cfg_override=BrokerCfg(commission_per_million=0),
    )
    pnl0 = pd.read_csv(out0 / "trades.csv")["pnl_usd"].sum()
    out1 = tmp_path / "o1"
    run_backtest_bt(
        str(m1),
        str(zones),
        symbol="EURUSD",
        htf="H1",
        outdir=str(out1),
        broker_cfg_override=BrokerCfg(commission_per_million=100),
    )
    pnl1 = pd.read_csv(out1 / "trades.csv")["pnl_usd"].sum()
    assert pnl0 > pnl1
