from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class VBTCfg:
    """Configuration holder for the light vectorbt bridge."""

    min_zone_grade: str = "B"
    zone_staleness_max_bars: int = 180
    use_trend_filter: bool = True
    require_nearby_sweep: bool = False
    touch_reject: Dict | None = None
    choch_bos_in_zone: Dict | None = None
    risk: Dict | None = None
    fees: Dict | None = None


# ---------------------------------------------------------------------------
# Context preparation
# ---------------------------------------------------------------------------


def prepare_context(
    m1_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    trend_timeline: Optional[pd.DataFrame],
    sweeps_df: Optional[pd.DataFrame],
    cfg: VBTCfg,
) -> pd.DataFrame:
    """Attach minimal zone context to the M1 dataframe.

    This simplified implementation merely attaches the first zone to all bars
    and computes a rolling ATR approximation. Real project would perform
    sophisticated nearest zone selection and filtering.
    """

    df = m1_df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("m1_df must contain high/low/close columns")

    # simple ATR approximation using high-low range
    window = cfg.risk.get("atr_window_m1", 14) if cfg.risk else 14
    df["atr_m1"] = df["high"].sub(df["low"]).rolling(window).mean()

    if not zones_df.empty:
        z = zones_df.iloc[0]
        df["zone_id"] = z.get("zone_id", 0)
        df["y_top"] = float(z.get("top", z.get("y_top", df["high"].max())))
        df["y_bottom"] = float(z.get("bottom", z.get("y_bottom", df["low"].min())))
        df["side"] = z.get("side", "long")
    else:
        df["zone_id"] = None
        df["y_top"] = float("nan")
        df["y_bottom"] = float("nan")
        df["side"] = None

    df["in_zone"] = (
        (df["low"] <= df["y_top"]) & (df["high"] >= df["y_bottom"])
    )
    return df


# ---------------------------------------------------------------------------
# Signal derivation
# ---------------------------------------------------------------------------


def derive_signals(
    m1_ctx: pd.DataFrame, cfg: VBTCfg
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Derive simplified entry signals from prepared context.

    The function looks for pre-marked columns ``touch_reject`` and
    ``choch_bos_in_zone`` on ``m1_ctx``. In real usage these would be derived
    from price action; for the unit tests we accept boolean hints.
    """

    entries_long = pd.Series(False, index=m1_ctx.index)
    entries_short = pd.Series(False, index=m1_ctx.index)
    meta_rows = []

    legs = []
    if cfg.risk and cfg.risk.get("multi_tp"):
        legs = cfg.risk["multi_tp"].get("legs", [])

    sl_pad_mult = cfg.risk.get("sl_pad_atr_mult", 0.0) if cfg.risk else 0.0

    for idx, row in m1_ctx.iterrows():
        if not row.get("in_zone", False):
            continue
        side = row.get("side", "long")
        trigger = None
        tr = row.get("touch_reject")
        if (
            cfg.touch_reject
            and cfg.touch_reject.get("enabled", False)
            and tr is True
        ):
            trigger = "touch_reject"
        if (
            trigger is None
            and cfg.choch_bos_in_zone
            and cfg.choch_bos_in_zone.get("enabled", False)
            and row.get("choch_bos_in_zone") is True
        ):
            trigger = "choch_bos_in_zone"
        if trigger is None:
            continue

        entry_price = row["close"]
        atr = row.get("atr_m1", 0.0)
        if pd.isna(atr):
            atr = 0.0
        if side == "long":
            sl = row["y_bottom"] - sl_pad_mult * atr
        else:
            sl = row["y_top"] + sl_pad_mult * atr
        risk = entry_price - sl if side == "long" else sl - entry_price
        if risk <= 0:
            continue

        if side == "long":
            entries_long.loc[idx] = True
        else:
            entries_short.loc[idx] = True

        leg_data = {}
        for i, leg in enumerate(legs, start=1):
            r_mult = float(leg.get("r", 1.0))
            tp = entry_price + r_mult * risk if side == "long" else entry_price - r_mult * risk
            leg_data[f"tp{i}"] = tp
            leg_data[f"r{i}"] = r_mult
            leg_data[f"w{i}"] = float(leg.get("weight", 0.0))

        meta_rows.append(
            {
                "index": idx,
                "zone_id": row.get("zone_id"),
                "trigger": trigger,
                "side": side,
                "entry": entry_price,
                "sl": sl,
                **leg_data,
            }
        )

    meta_df = pd.DataFrame(meta_rows).set_index("index") if meta_rows else pd.DataFrame()
    return entries_long, entries_short, meta_df


# ---------------------------------------------------------------------------
# Backtest execution (simplified)
# ---------------------------------------------------------------------------


def run_vectorbt(
    m1_df: pd.DataFrame,
    entries_long: pd.Series,
    entries_short: pd.Series,
    meta_df: pd.DataFrame,
    cfg: VBTCfg,
    initial_equity: float,
) -> Dict[str, pd.DataFrame]:
    """Run a very small backtest engine using price scanning.

    This is not a full vectorbt integration but mimics its output structure
    sufficiently for unit testing and rapid prototyping.
    """

    if meta_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "trade_id",
                "leg",
                "side",
                "zone_id",
                "trigger",
                "entry_time",
                "entry",
                "sl",
                "tp",
                "exit_time",
                "exit",
                "pnl_pips",
                "pnl_R",
                "fees_usd",
                "tags_json",
            ]
        )
        equity_df = pd.DataFrame(
            {
                "time": m1_df.index,
                "equity": initial_equity,
                "drawdown": 0.0,
                "open_legs": 0,
            }
        )
        return {"trades": trades_df, "equity_curve": equity_df}

    risk_cfg = cfg.risk or {}
    pip_size = risk_cfg.get("pip_size", 0.0001)
    risk_pct = risk_cfg.get("risk_fixed_pct", 1.0) / 100.0
    multi_legs = risk_cfg.get("multi_tp", {}).get("legs", [])
    risk_amount = initial_equity * risk_pct
    pip_value = risk_cfg.get("contract_size", 100000) * pip_size

    trades = []

    for idx, meta in meta_df.iterrows():
        entry_price = meta["entry"]
        sl = meta["sl"]
        side = meta["side"]
        entry_loc = m1_df.index.get_loc(idx)
        risk_pips = abs((entry_price - sl) / pip_size)
        for leg_no, leg in enumerate(multi_legs, start=1):
            tp = meta.get(f"tp{leg_no}")
            r_mult = meta.get(f"r{leg_no}")
            weight = meta.get(f"w{leg_no}")
            exit_price = tp
            exit_idx = None
            for k in range(entry_loc + 1, len(m1_df)):
                bar = m1_df.iloc[k]
                if side == "long":
                    if bar["low"] <= sl:
                        exit_price = sl
                        r_res = -1.0
                        exit_idx = k
                        break
                    if bar["high"] >= tp:
                        exit_price = tp
                        r_res = r_mult
                        exit_idx = k
                        break
                else:
                    if bar["high"] >= sl:
                        exit_price = sl
                        r_res = -1.0
                        exit_idx = k
                        break
                    if bar["low"] <= tp:
                        exit_price = tp
                        r_res = r_mult
                        exit_idx = k
                        break
            if exit_idx is None:
                exit_idx = len(m1_df) - 1
                exit_price = m1_df.iloc[exit_idx]["close"]
                r_res = 0.0
            exit_time = m1_df.index[exit_idx]
            pnl_pips = r_res * risk_pips
            pnl_R = r_res * weight
            notional = (risk_amount * weight) / (risk_pips * pip_value)
            fees_usd = 0.0
            trades.append(
                {
                    "trade_id": idx,
                    "leg": leg_no,
                    "side": side,
                    "zone_id": meta.get("zone_id"),
                    "trigger": meta.get("trigger"),
                    "entry_time": idx,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "exit_time": exit_time,
                    "exit": exit_price,
                    "pnl_pips": pnl_pips,
                    "pnl_R": pnl_R,
                    "fees_usd": fees_usd,
                    "tags_json": "{}",
                }
            )

    trades_df = pd.DataFrame(trades)

    # Build equity curve
    equity_series = pd.Series(initial_equity, index=m1_df.index)
    open_legs = pd.Series(0, index=m1_df.index)
    eq = initial_equity
    open_count = 0
    for t in m1_df.index:
        # open legs at this time
        for _ in trades_df[trades_df["entry_time"] == t].itertuples():
            open_count += 1
        # close legs at this time
        for tr in trades_df[trades_df["exit_time"] == t].itertuples():
            eq += tr.pnl_R * risk_amount
            open_count -= 1
        equity_series.loc[t] = eq
        open_legs.loc[t] = open_count
    drawdown = equity_series / equity_series.cummax() - 1.0
    equity_df = pd.DataFrame(
        {
            "time": equity_series.index,
            "equity": equity_series.values,
            "drawdown": drawdown.values,
            "open_legs": open_legs.values,
        }
    )

    return {"trades": trades_df, "equity_curve": equity_df}
