from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import backtrader as bt
import yaml

from .bt_broker import BrokerCfg, build_broker_and_sizers
from .bt_strategy import POIExecutionStrategy, StratCfg
from .metrics import summarize_bt


@dataclass
class BTRunCfg:
    start: Optional[str] = None
    end: Optional[str] = None
    initial_equity: float = 10000.0
    profile: str = "bt_m1"


def _load_profile(profile: str) -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "backtest.yml"
    data: dict = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    return data.get("profiles", {}).get(profile, {})


def run_backtest_bt(
    m1_parquet: str,
    zones_csv: str,
    symbol: str,
    htf: str,
    outdir: str,
    trend_timeline_csv: Optional[str] = None,
    sweeps_csv: Optional[str] = None,
    asia_daily_csv: Optional[str] = None,
    eq_clusters_csv: Optional[str] = None,
    profile: str = "bt_m1",
    broker_cfg_override: Optional[BrokerCfg] = None,
) -> dict:
    """Run simple backtrader based backtest."""

    profile_cfg = _load_profile(profile)
    risk_cfg = profile_cfg.get("risk", {})
    fees_cfg = profile_cfg.get("fees", {})
    strat_cfg = StratCfg(
        allow_sessions=profile_cfg.get("allow_sessions", []),
        session_hours_utc=profile_cfg.get("session_hours_utc", {}),
        min_minutes_between_trades=profile_cfg.get("min_minutes_between_trades", 0),
        max_concurrent_trades=profile_cfg.get("max_concurrent_trades", 1),
        one_trade_per_zone=profile_cfg.get("one_trade_per_zone", False),
        use_trend_filter=profile_cfg.get("use_trend_filter", False),
        min_zone_grade=profile_cfg.get("min_zone_grade", "B"),
        zone_staleness_max_bars=profile_cfg.get("zone_staleness_max_bars", 0),
        require_nearby_sweep=profile_cfg.get("require_nearby_sweep", False),
        triggers=profile_cfg.get("triggers", {}),
        risk=risk_cfg,
        risk_caps=profile_cfg.get("risk_caps", {}),
    )

    broker_cfg = broker_cfg_override or BrokerCfg(
        spread_pips=fees_cfg.get("spread_pips", 0.0),
        slippage_pips=fees_cfg.get("slippage_pips", 0.0),
        commission_per_million=fees_cfg.get("commission_per_million", 0.0),
        comm_apply_both_sides=fees_cfg.get("comm_apply_both_sides", True),
        pip_size=risk_cfg.get("pip_size", 0.0001),
        contract_size=risk_cfg.get("contract_size", 100000),
    )

    m1_df = pd.read_parquet(m1_parquet)
    zones_df = pd.read_csv(zones_csv)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(profile_cfg.get("initial_equity", 10000.0))
    build_broker_and_sizers(cerebro, broker_cfg)
    data_feed = bt.feeds.PandasData(dataname=m1_df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(POIExecutionStrategy, cfg=strat_cfg, zones=zones_df)
    strat = cerebro.run()[0]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(strat.trade_logs)
    fills_df = pd.DataFrame(strat.fill_logs)
    equity_df = pd.DataFrame(strat.equity_logs)
    trades_df.to_csv(out_path / "trades.csv", index=False)
    fills_df.to_csv(out_path / "fills.csv", index=False)
    equity_df.to_csv(out_path / "equity_curve.csv", index=False)
    summary = summarize_bt(trades_df, equity_df)
    (out_path / "bt_summary.json").write_text(pd.Series(summary).to_json())
    return summary
