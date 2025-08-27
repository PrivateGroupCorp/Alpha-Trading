from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import pandas as pd

from alpha.core.indicators import atr
from .triggers import (
    TriggerCfgTouch,
    TriggerCfgChoch,
    check_touch_reject,
    check_choch_in_zone,
)
from .risk import (
    RiskCfg,
    compute_sl_tp,
    position_size,
    fees_and_friction,
)


@dataclass
class ExecCfg:
    allow_sessions: List[str]
    session_hours_utc: Dict[str, List[int]]
    min_minutes_between_trades: int
    max_concurrent_trades: int
    one_trade_per_zone: bool
    use_trend_filter: bool
    min_zone_grade: str
    zone_staleness_max_bars: int
    require_nearby_sweep: bool
    triggers: Dict[str, Dict[str, float]]
    risk: Dict[str, object]
    risk_caps: Dict[str, float]


_SESSION_DEFAULTS = {
    "AS": (0, 8),
    "EU": (7, 16),
    "US": (12, 21),
}


@dataclass
class _OpenTrade:
    trade_id: int
    side: str
    zone_id: int
    trigger: str
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    size_units: float
    remaining: float
    sl: float
    tps: List[float]
    tp_pcts: List[float]
    tp_hits: List[bool]
    risk_per_unit: float
    risk_R: float
    tags: Dict[str, object]


_DEF_TRIGGER_CFG = {
    "touch_reject": TriggerCfgTouch(),
    "choch_bos_in_zone": TriggerCfgChoch(),
}


_GRADES = {"A": 3, "B": 2, "C": 1, "D": 0}


def _grade_ok(grade: str, min_grade: str) -> bool:
    return _GRADES.get(str(grade), -1) >= _GRADES.get(str(min_grade), -1)


def _in_session(ts: pd.Timestamp, allow: List[str], hours_map: Dict[str, List[int]]) -> bool:
    h = ts.hour
    for sess in allow:
        start, end = hours_map.get(sess, _SESSION_DEFAULTS.get(sess, (0, 24)))
        if start <= h < end:
            return True
    return False


def run_execution(
    m1_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    trend_timeline: Optional[pd.DataFrame],
    sweeps_df: Optional[pd.DataFrame],
    context: Dict,
    exec_cfg: ExecCfg,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    initial_equity_usd: float = 10000.0,
) -> Dict[str, pd.DataFrame]:
    """Run simplified execution engine on M1 data."""

    df = m1_df.copy()
    if start_time is not None:
        df = df[df.index >= start_time]
    if end_time is not None:
        df = df[df.index <= end_time]
    df = df.sort_index()

    # compute ATR
    atr_series = atr(df, window=int(exec_cfg.risk.get("atr_window_m1", 14)))

    equity = initial_equity_usd
    open_trades: List[_OpenTrade] = []
    trade_records = []
    fills = []
    used_zones = set()
    trade_id_counter = 1
    last_trade_time = pd.Timestamp("1970-01-01", tz="UTC")

    daily_R = {}
    weekly_R = {}
    consec_losses = 0

    tp_schema = exec_cfg.risk.get("tp_schema", [{"r": 1.0, "pct": 1.0}])
    breakeven_after = float(exec_cfg.risk.get("breakeven_after", 1.0))
    trailing_mode = exec_cfg.risk.get("trailing", {}).get("mode", "off")

    # iterate over bars
    for j, (ts, bar) in enumerate(df.iterrows()):
        # update open trades
        for trade in list(open_trades):
            high = bar["high"]
            low = bar["low"]
            # compute R moves
            if trade.side == "long":
                mae = (trade.entry_price - low) / trade.risk_per_unit
                mfe = (high - trade.entry_price) / trade.risk_per_unit
            else:
                mae = (high - trade.entry_price) / trade.risk_per_unit
                mfe = (trade.entry_price - low) / trade.risk_per_unit
            trade.tags["mae_R"] = min(trade.tags.get("mae_R", 0.0), mae)
            trade.tags["mfe_R"] = max(trade.tags.get("mfe_R", 0.0), mfe)

            # trailing not implemented beyond breakeven
            if trade.side == "long":
                # check TP
                while (
                    trade.tp_hits.count(False) > 0
                    and high >= trade.tps[trade.tp_hits.index(False)]
                ):
                    idx = trade.tp_hits.index(False)
                    pct = trade.tp_pcts[idx]
                    units_close = trade.size_units * pct
                    trade.remaining -= units_close
                    trade.tp_hits[idx] = True
                    profit_per_unit = trade.tps[idx] - trade.entry_price
                    pnl = profit_per_unit * units_close
                    equity += pnl
                    fills.append(
                        {
                            "trade_id": trade.trade_id,
                            "time": ts,
                            "kind": f"tp{idx+1}",
                            "price": trade.tps[idx],
                            "reason": "tp",
                        }
                    )
                # breakeven move
                if trade.sl < trade.entry_price and mfe >= breakeven_after:
                    trade.sl = trade.entry_price
                # stop loss
                if low <= trade.sl:
                    pnl = (trade.sl - trade.entry_price) * trade.remaining
                    equity += pnl
                    fills.append(
                        {
                            "trade_id": trade.trade_id,
                            "time": ts,
                            "kind": "sl",
                            "price": trade.sl,
                            "reason": "sl",
                        }
                    )
                    trade.tags["exit_time"] = ts
                    trade.tags["exit_price"] = trade.sl
                    trade.tags["pnl_usd"] = pnl
                    trade.tags["pnl_R"] = pnl / (trade.risk_per_unit * trade.size_units)
                    if trade.tags["pnl_R"] < 0:
                        consec_losses += 1
                    else:
                        consec_losses = 0
                    # update risk caps
                    day = ts.date()
                    week = ts.isocalendar().week
                    daily_R[day] = daily_R.get(day, 0.0) + trade.tags["pnl_R"]
                    weekly_R[week] = weekly_R.get(week, 0.0) + trade.tags["pnl_R"]
                    trade_records.append(trade.tags)
                    open_trades.remove(trade)
            else:  # short
                while (
                    trade.tp_hits.count(False) > 0
                    and low <= trade.tps[trade.tp_hits.index(False)]
                ):
                    idx = trade.tp_hits.index(False)
                    pct = trade.tp_pcts[idx]
                    units_close = trade.size_units * pct
                    trade.remaining -= units_close
                    trade.tp_hits[idx] = True
                    profit_per_unit = trade.entry_price - trade.tps[idx]
                    pnl = profit_per_unit * units_close
                    equity += pnl
                    fills.append(
                        {
                            "trade_id": trade.trade_id,
                            "time": ts,
                            "kind": f"tp{idx+1}",
                            "price": trade.tps[idx],
                            "reason": "tp",
                        }
                    )
                if trade.sl > trade.entry_price and mfe >= breakeven_after:
                    trade.sl = trade.entry_price
                if high >= trade.sl:
                    pnl = (trade.entry_price - trade.sl) * trade.remaining
                    equity += pnl
                    fills.append(
                        {
                            "trade_id": trade.trade_id,
                            "time": ts,
                            "kind": "sl",
                            "price": trade.sl,
                            "reason": "sl",
                        }
                    )
                    trade.tags["exit_time"] = ts
                    trade.tags["exit_price"] = trade.sl
                    trade.tags["pnl_usd"] = pnl
                    trade.tags["pnl_R"] = pnl / (trade.risk_per_unit * trade.size_units)
                    if trade.tags["pnl_R"] < 0:
                        consec_losses += 1
                    else:
                        consec_losses = 0
                    day = ts.date()
                    week = ts.isocalendar().week
                    daily_R[day] = daily_R.get(day, 0.0) + trade.tags["pnl_R"]
                    weekly_R[week] = weekly_R.get(week, 0.0) + trade.tags["pnl_R"]
                    trade_records.append(trade.tags)
                    open_trades.remove(trade)

        # check if can open new trade
        if len(open_trades) >= exec_cfg.max_concurrent_trades:
            continue
        if (ts - last_trade_time).total_seconds() / 60.0 < exec_cfg.min_minutes_between_trades:
            continue
        if not _in_session(ts, exec_cfg.allow_sessions, exec_cfg.session_hours_utc):
            continue

        day = ts.date()
        week = ts.isocalendar().week
        if daily_R.get(day, 0.0) <= -exec_cfg.risk_caps.get("max_daily_loss_R", float("inf")):
            continue
        if weekly_R.get(week, 0.0) <= -exec_cfg.risk_caps.get("max_weekly_loss_R", float("inf")):
            continue
        if consec_losses >= exec_cfg.risk_caps.get("stop_after_consecutive_losses", 999):
            continue

        bar = df.iloc[j]
        atr_val = float(atr_series.iloc[j])

        # candidate zones that price is within
        for _, zone in zones_df.iterrows():
            zone_id = int(zone.get("zone_id", zone.get("id", 0)))
            if exec_cfg.one_trade_per_zone and zone_id in used_zones:
                continue
            if not _grade_ok(zone.get("grade", "A"), exec_cfg.min_zone_grade):
                continue
            if j - int(zone.get("break_idx", 0)) > exec_cfg.zone_staleness_max_bars:
                continue
            side = zone.get("side", "long")
            top = float(zone["y_top"])
            bottom = float(zone["y_bottom"])
            touched = (
                bar["low"] <= top and bar["high"] >= bottom
            ) if side == "long" else (bar["high"] >= bottom and bar["low"] <= top)
            if not touched:
                continue

            trigger_hit = None
            trg_touch = exec_cfg.triggers.get("touch_reject", {})
            if trg_touch.get("enabled", True):
                cfg_dict = {k: v for k, v in trg_touch.items() if k != "enabled"}
                cfg = TriggerCfgTouch(**cfg_dict)
                if check_touch_reject(bar, zone, atr_val, cfg):
                    trigger_hit = "touch_reject"
            trg_choch = exec_cfg.triggers.get("choch_bos_in_zone", {})
            if (
                trigger_hit is None
                and trg_choch.get("enabled", True)
            ):
                cfgc_dict = {k: v for k, v in trg_choch.items() if k != "enabled"}
                cfgc = TriggerCfgChoch(**cfgc_dict)
                if check_choch_in_zone(df, j, zone, atr_series, cfgc):
                    trigger_hit = "choch_bos_in_zone"

            if trigger_hit:
                # open trade
                direction = "long" if side == "long" else "short"
                sl_tp = compute_sl_tp(
                    bar["close"],
                    zone,
                    atr_val,
                    direction,
                    tp_schema,
                    exec_cfg.risk.get("pip_size", 0.0001),
                    exec_cfg.risk.get("sl_pad_atr_mult", 0.1),
                )
                size = position_size(
                    equity,
                    exec_cfg.risk.get("risk_fixed_pct", 0.5),
                    sl_tp["risk_per_unit"],
                )
                trade = _OpenTrade(
                    trade_id=trade_id_counter,
                    side=direction,
                    zone_id=zone_id,
                    trigger=trigger_hit,
                    entry_idx=j,
                    entry_time=ts,
                    entry_price=bar["close"],
                    size_units=size,
                    remaining=size,
                    sl=sl_tp["sl"],
                    tps=sl_tp["tps"],
                    tp_pcts=[tp.get("pct", 1.0) for tp in tp_schema],
                    tp_hits=[False] * len(tp_schema),
                    risk_per_unit=sl_tp["risk_per_unit"],
                    risk_R=exec_cfg.risk.get("risk_fixed_pct", 0.5) / 100.0,
                    tags={
                        "trade_id": trade_id_counter,
                        "side": direction,
                        "zone_id": zone_id,
                        "trigger": trigger_hit,
                        "entry_time": ts,
                        "entry_idx": j,
                        "entry_price": bar["close"],
                        "sl_init": sl_tp["sl"],
                    },
                )
                open_trades.append(trade)
                last_trade_time = ts
                used_zones.add(zone_id)
                fills.append(
                    {
                        "trade_id": trade.trade_id,
                        "time": ts,
                        "kind": "entry",
                        "price": bar["close"],
                        "reason": trigger_hit,
                    }
                )
                trade_id_counter += 1
                break

    trades_df = pd.DataFrame(trade_records)
    fills_df = pd.DataFrame(fills)
    equity_curve = pd.DataFrame(
        {"time": df.index, "equity": equity}, index=df.index
    ).reset_index(drop=True)

    summary = {
        "n_trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl_R"] > 0).mean()) if len(trades_df) else 0.0,
        "avg_R": float(trades_df["pnl_R"].mean()) if len(trades_df) else 0.0,
        "max_dd_R": float(min(daily_R.values())) if daily_R else 0.0,
    }

    summary_df = pd.DataFrame([summary])

    return {
        "trades": trades_df,
        "fills": fills_df,
        "equity_curve": equity_curve,
        "summary": summary_df,
    }
