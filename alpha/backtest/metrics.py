from __future__ import annotations

from typing import Dict

import pandas as pd


def summarize_bt(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    """Summarize backtest trades and equity into KPI dictionary."""

    summary: Dict[str, object] = {}
    if trades_df.empty:
        summary.update(
            {
                "n_trades": 0,
                "n_legs": 0,
                "win_rate_legs": 0.0,
                "win_rate_trades": 0.0,
                "avg_R": 0.0,
                "exp_R": 0.0,
                "max_dd_R": 0.0,
                "sharpe_like": 0.0,
                "by_trigger": {},
                "by_grade": {},
                "by_session": {},
                "params": {},
            }
        )
        return summary

    trades_df = trades_df.copy()
    trades_df["win"] = trades_df["pnl_R"] > 0
    trades_df["entry_time"] = pd.to_datetime(trades_df.get("entry_time"), errors="coerce")

    summary["n_trades"] = int(trades_df["trade_id"].nunique())
    summary["n_legs"] = int(len(trades_df))
    summary["win_rate_legs"] = float(trades_df["win"].mean())
    trade_win = trades_df.groupby("trade_id")["pnl_R"].sum() > 0
    summary["win_rate_trades"] = float(trade_win.mean())
    summary["avg_R"] = float(trades_df["pnl_R"].mean())
    summary["exp_R"] = float(trades_df["pnl_R"].mean())
    summary["max_dd_R"] = float(equity_df["drawdown"].min())
    if trades_df["pnl_R"].std() != 0:
        summary["sharpe_like"] = float(
            trades_df["pnl_R"].mean() / trades_df["pnl_R"].std()
        )
    else:
        summary["sharpe_like"] = 0.0

    summary["by_trigger"] = (
        trades_df.groupby("trigger")["pnl_R"].sum().to_dict()
    )
    if "grade" in trades_df.columns:
        summary["by_grade"] = trades_df.groupby("grade")["pnl_R"].sum().to_dict()
    else:
        summary["by_grade"] = {}

    sessions = pd.cut(
        trades_df["entry_time"].dt.hour,
        bins=[-1, 7, 15, 23],
        labels=["asia", "london", "ny"],
    )
    summary["by_session"] = trades_df.groupby(sessions)["pnl_R"].sum().to_dict()
    summary["params"] = {}
    return summary
