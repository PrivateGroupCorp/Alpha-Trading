"""Backtesting utilities bridging vectorbt."""

from .vbt_bridge import VBTCfg, prepare_context, derive_signals, run_vectorbt
from .metrics import summarize_bt
from .bt_runner import run_backtest_bt
from .bt_broker import BrokerCfg
from .bt_strategy import StratCfg, POIExecutionStrategy

__all__ = [
    "VBTCfg",
    "prepare_context",
    "derive_signals",
    "run_vectorbt",
    "summarize_bt",
    "run_backtest_bt",
    "BrokerCfg",
    "StratCfg",
    "POIExecutionStrategy",
]
