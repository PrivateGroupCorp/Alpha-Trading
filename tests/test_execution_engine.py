import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.execution.engine import ExecCfg, run_execution
from alpha.execution.risk import fees_and_friction


def _make_base_df(n=20, price=100.0):
    times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": price,
        "high": price + 0.05,
        "low": price - 0.05,
        "close": price,
    }, index=times)
    return df


def _default_cfg(**kwargs):
    cfg = ExecCfg(
        allow_sessions=["AS", "EU", "US"],
        session_hours_utc={"AS": [0,24], "EU": [0,24], "US": [0,24]},
        min_minutes_between_trades=0,
        max_concurrent_trades=2,
        one_trade_per_zone=True,
        use_trend_filter=False,
        min_zone_grade="B",
        zone_staleness_max_bars=1000,
        require_nearby_sweep=False,
        triggers={
            "touch_reject": {"enabled": True},
            "choch_bos_in_zone": {"enabled": True, "lookahead_bars": 3},
        },
        risk={
            "risk_fixed_pct": 1.0,
            "atr_window_m1": 3,
            "sl_pad_atr_mult": 0.1,
            "tp_schema": [{"r": 1.0, "pct": 0.5}, {"r": 2.0, "pct": 0.5}],
            "breakeven_after": 1.0,
        },
        risk_caps={
            "max_daily_loss_R": 100,
            "max_weekly_loss_R": 100,
            "stop_after_consecutive_losses": 3,
        },
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_touch_reject_trade_be():
    df = _make_base_df()
    # bar triggering touch reject
    df.iloc[5] = [100.03, 100.05, 100.01, 100.04]
    df.iloc[6] = [100.05, 100.15, 100.04, 100.10]
    df.iloc[7] = [100.05, 100.06, 99.90, 99.95]

    zones = pd.DataFrame([
        {
            "zone_id": 1,
            "side": "long",
            "y_top": 100.02,
            "y_bottom": 99.98,
            "grade": "A",
            "break_idx": 0,
        }
    ])

    cfg = _default_cfg()
    cfg.triggers["choch_bos_in_zone"]["enabled"] = False
    res = run_execution(df, zones, None, None, {}, cfg)
    trades = res["trades"]
    fills = res["fills"]
    assert len(trades) == 1
    assert list(fills["kind"]) == ["entry", "tp1", "sl"]
    sl_price = float(fills[fills["kind"] == "sl"]["price"].iloc[0])
    entry_price = float(trades["entry_price"].iloc[0])
    assert abs(sl_price - entry_price) < 1e-6
    assert trades["trigger"].iloc[0] == "touch_reject"


def test_choch_trigger():
    df = _make_base_df()
    df.iloc[5] = [100.03, 100.06, 100.00, 100.04]
    df.iloc[6] = [100.02, 100.03, 99.95, 99.97]
    df.iloc[7] = [99.98, 100.40, 99.97, 100.30]
    df.iloc[8] = [100.10, 100.15, 99.79, 99.90]

    zones = pd.DataFrame([
        {
            "zone_id": 1,
            "side": "long",
            "y_top": 100.05,
            "y_bottom": 99.95,
            "grade": "A",
            "break_idx": 0,
        }
    ])

    cfg = _default_cfg()
    cfg.triggers["touch_reject"]["enabled"] = False
    cfg.triggers["choch_bos_in_zone"]["enabled"] = True
    res = run_execution(df, zones, None, None, {}, cfg)
    trades = res["trades"]
    assert len(trades) == 1
    assert trades["trigger"].iloc[0] == "choch_bos_in_zone"


def test_fees_and_friction():
    res = fees_and_friction(1_000_000, 1.0, 2.0, 50.0, 0.0001)
    assert abs(res["spread_cost_usd"] - 0.0001) < 1e-12
    assert abs(res["slippage_cost_usd"] - 0.0002) < 1e-12
    assert abs(res["commission_usd"] - 50.0) < 1e-12


def test_risk_cap_consecutive_losses():
    df = _make_base_df(30)
    # four potential trades
    entry_pattern = [100.03, 100.05, 100.01, 100.04]
    loss_pattern = [100.02, 100.03, 99.70, 99.80]
    for k in range(4):
        start = 1 + k * 2
        df.iloc[start] = entry_pattern
        df.iloc[start + 1] = loss_pattern

    zones = pd.DataFrame([
        {
            "zone_id": 1,
            "side": "long",
            "y_top": 100.02,
            "y_bottom": 99.98,
            "grade": "A",
            "break_idx": 0,
        }
    ])

    cfg = _default_cfg(
        one_trade_per_zone=False,
        triggers={"touch_reject": {"enabled": True}, "choch_bos_in_zone": {"enabled": False}},
    )
    cfg.min_minutes_between_trades = 0
    cfg.risk_caps["stop_after_consecutive_losses"] = 3

    res = run_execution(df, zones, None, None, {}, cfg)
    trades = res["trades"]
    assert len(trades) == 3
