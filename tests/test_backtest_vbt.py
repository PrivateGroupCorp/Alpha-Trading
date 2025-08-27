import pandas as pd

from alpha.backtest.vbt_bridge import VBTCfg, prepare_context, derive_signals, run_vectorbt
from alpha.backtest.metrics import summarize_bt


def _make_price_df():
    idx = pd.date_range("2020-01-01", periods=5, freq="T", tz="UTC")
    data = {
        "open": [1.0005, 1.0010, 1.0015, 1.0020, 1.0025],
        "high": [1.0010, 1.0015, 1.0020, 1.0025, 1.0030],
        "low": [1.0000, 1.0005, 1.0010, 1.0015, 1.0020],
        "close": [1.0005, 1.0010, 1.0015, 1.0020, 1.0025],
    }
    return pd.DataFrame(data, index=idx)


def _make_zone_df():
    return pd.DataFrame(
        {
            "zone_id": [1],
            "top": [1.0010],
            "bottom": [1.0000],
            "side": ["long"],
        }
    )


def test_touch_reject_long_multi_leg():
    df = _make_price_df()
    zones = _make_zone_df()
    cfg = VBTCfg(
        touch_reject={"enabled": True},
        choch_bos_in_zone={"enabled": False},
        risk={
            "pip_size": 0.0001,
            "sl_pad_atr_mult": 0.0,
            "multi_tp": {
                "legs": [
                    {"r": 1.0, "weight": 0.30},
                    {"r": 2.0, "weight": 0.40},
                    {"r": 3.0, "weight": 0.30},
                ]
            },
        },
    )
    ctx = prepare_context(df, zones, None, None, cfg)
    ctx.loc[df.index[0], "touch_reject"] = True
    entries_long, entries_short, meta = derive_signals(ctx, cfg)
    result = run_vectorbt(df, entries_long, entries_short, meta, cfg, 10000.0)
    trades = result["trades"]
    assert len(trades) == 3
    assert trades["pnl_R"].round(2).tolist() == [0.30, 0.80, 0.90]
    summary = summarize_bt(trades, result["equity_curve"])
    assert summary["n_trades"] == 1
    assert summary["n_legs"] == 3


def test_leg_weights_change_pnl():
    df = _make_price_df()
    zones = _make_zone_df()
    base_ctx = prepare_context(df, zones, None, None, VBTCfg())
    base_ctx.loc[df.index[0], "touch_reject"] = True

    cfg1 = VBTCfg(
        touch_reject={"enabled": True},
        risk={
            "pip_size": 0.0001,
            "sl_pad_atr_mult": 0.0,
            "multi_tp": {"legs": [{"r": 1.0, "weight": 0.5}, {"r": 2.0, "weight": 0.5}]},
        },
    )
    e1_l, e1_s, m1 = derive_signals(base_ctx, cfg1)
    res1 = run_vectorbt(df, e1_l, e1_s, m1, cfg1, 10000.0)
    total1 = res1["trades"]["pnl_R"].sum()

    cfg2 = VBTCfg(
        touch_reject={"enabled": True},
        risk={
            "pip_size": 0.0001,
            "sl_pad_atr_mult": 0.0,
            "multi_tp": {"legs": [{"r": 1.0, "weight": 0.2}, {"r": 2.0, "weight": 0.8}]},
        },
    )
    e2_l, e2_s, m2 = derive_signals(base_ctx, cfg2)
    res2 = run_vectorbt(df, e2_l, e2_s, m2, cfg2, 10000.0)
    total2 = res2["trades"]["pnl_R"].sum()

    assert total1 != total2


def test_trigger_toggle_changes_signals():
    df = _make_price_df()
    zones = _make_zone_df()
    ctx = prepare_context(df, zones, None, None, VBTCfg())
    ctx.loc[df.index[0], "choch_bos_in_zone"] = True

    cfg = VBTCfg(
        touch_reject={"enabled": False},
        choch_bos_in_zone={"enabled": True},
        risk={
            "pip_size": 0.0001,
            "sl_pad_atr_mult": 0.0,
            "multi_tp": {"legs": [{"r": 1.0, "weight": 1.0}]},
        },
    )
    entries_long, _, _ = derive_signals(ctx, cfg)
    assert entries_long.sum() == 1
    cfg.choch_bos_in_zone["enabled"] = False
    entries_long2, _, _ = derive_signals(ctx, cfg)
    assert entries_long2.sum() == 0
