from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import backtrader as bt
import pandas as pd

from .bt_broker import BrokerCfg


@dataclass
class StratCfg:
    allow_sessions: List[str]
    session_hours_utc: Dict[str, List[int]]
    min_minutes_between_trades: int
    max_concurrent_trades: int
    one_trade_per_zone: bool
    use_trend_filter: bool
    min_zone_grade: str
    zone_staleness_max_bars: int
    require_nearby_sweep: bool
    triggers: Dict
    risk: Dict
    risk_caps: Dict


class POIExecutionStrategy(bt.Strategy):
    params = (
        ("cfg", None),
        ("zones", None),
    )

    def __init__(self) -> None:
        self.cfg: StratCfg = self.params.cfg
        self.zones: pd.DataFrame = self.params.zones if self.params.zones is not None else pd.DataFrame()
        self.zones = self.zones.copy()
        self.order_meta: Dict[int, Dict] = {}
        self.trade_logs: List[Dict] = []
        self.fill_logs: List[Dict] = []
        self.equity_logs: List[Dict] = []
        self.trade_seq = 1

    # --- helpers -----------------------------------------------------
    def _calc_size(self, weight: float) -> float:
        return max(int(round(weight * 100)), 1)

    def _build_brackets(
        self, side: str, entry_price: float, sl: float, tps: List[float], weights: List[float], meta: Dict
    ) -> None:
        for i, (tp, w) in enumerate(zip(tps, weights), start=1):
            size = self._calc_size(w)
            if side == "long":
                parent, stop, limit = self.buy_bracket(
                    size=size,
                    price=entry_price,
                    stopprice=sl,
                    limitprice=tp,
                    exectype=bt.Order.Market,
                )
            else:
                parent, stop, limit = self.sell_bracket(
                    size=size,
                    price=entry_price,
                    stopprice=sl,
                    limitprice=tp,
                    exectype=bt.Order.Market,
                )
            info = dict(meta)
            info.update({"leg": i, "side": side, "tp": tp, "sl": sl})
            parent.addinfo(**info)
            stop.addinfo(**info, order_role="sl")
            limit.addinfo(**info, order_role="tp")
            self.order_meta[parent.ref] = info
        self.trade_seq += 1

    # --- core --------------------------------------------------------
    def next(self) -> None:  # pragma: no cover - heavy on backtrader internals
        dt = bt.num2date(self.data.datetime[0])
        equity = self.broker.getvalue()
        self.equity_logs.append(
            {
                "time": dt,
                "equity": equity,
                "dd_R": 0.0,
                "drawdown": 0.0,
                "open_legs": len(self.order_meta),
            }
        )
        if len(self.order_meta) >= self.cfg.max_concurrent_trades:
            return
        if self.zones.empty:
            return
        price = self.data.close[0]
        zone = self.zones.iloc[0]
        if zone["kind"].lower() == "demand":
            if zone["price_bottom"] <= price <= zone["price_top"]:
                entry = price
                sl = zone["price_bottom"]
                dist = entry - sl
                tps = [entry + dist * leg["r"] for leg in self.cfg.risk.get("legs", [])]
                weights = [leg["weight"] for leg in self.cfg.risk.get("legs", [])]
                meta = {"trade_id": self.trade_seq, "zone_id": zone.get("zone_id", 0), "trigger": "touch"}
                self._build_brackets("long", entry, sl, tps, weights, meta)
                self.zones = self.zones.iloc[1:]
        else:
            if zone["price_bottom"] <= price <= zone["price_top"]:
                entry = price
                sl = zone["price_top"]
                dist = sl - entry
                tps = [entry - dist * leg["r"] for leg in self.cfg.risk.get("legs", [])]
                weights = [leg["weight"] for leg in self.cfg.risk.get("legs", [])]
                meta = {"trade_id": self.trade_seq, "zone_id": zone.get("zone_id", 0), "trigger": "touch"}
                self._build_brackets("short", entry, sl, tps, weights, meta)
                self.zones = self.zones.iloc[1:]

    # --- notifications -----------------------------------------------
    def notify_order(self, order):  # pragma: no cover - handled in bt
        if order.status in [order.Completed, order.Partial]:
            dt = bt.num2date(order.executed.dt)
            self.fill_logs.append(
                {
                    "time": dt,
                    "ref": order.ref,
                    "price": order.executed.price,
                    "size": order.executed.size,
                    "order_role": order.info.get("order_role", "entry"),
                    "trade_id": order.info.get("trade_id"),
                    "leg": order.info.get("leg"),
                    "side": order.info.get("side"),
                }
            )
            if order.info.get("order_role", "entry") == "entry":
                meta = self.order_meta.get(order.ref)
                if meta is not None:
                    meta["entry_time"] = dt
                    meta["entry"] = order.executed.price

    def notify_trade(self, trade):  # pragma: no cover - handled in bt
        if not trade.isclosed:
            return
        meta = self.order_meta.get(trade.ref, {})
        entry = meta.get("entry", trade.price)
        pnl_usd = trade.pnlcomm
        size = trade.size
        if meta.get("side") == "long":
            exit_price = entry + pnl_usd / size if size != 0 else entry
            pnl_pips = pnl_usd / (self.cfg.risk["pip_size"] * self.cfg.risk["contract_size"])
            risk_pips = (entry - meta.get("sl", entry)) / self.cfg.risk["pip_size"]
        else:
            exit_price = entry - pnl_usd / abs(size) if size != 0 else entry
            pnl_pips = pnl_usd / (self.cfg.risk["pip_size"] * self.cfg.risk["contract_size"])
            risk_pips = (meta.get("sl", entry) - entry) / self.cfg.risk["pip_size"]
        pnl_R = pnl_pips / risk_pips if risk_pips else 0.0
        self.trade_logs.append(
            {
                "trade_id": meta.get("trade_id", 0),
                "leg": meta.get("leg", 0),
                "side": meta.get("side"),
                "zone_id": meta.get("zone_id"),
                "trigger": meta.get("trigger"),
                "entry_time": meta.get("entry_time"),
                "entry": entry,
                "sl_init": meta.get("sl"),
                "sl_final": meta.get("sl"),
                "tp": meta.get("tp"),
                "exit_time": bt.num2date(self.data.datetime[0]),
                "exit": exit_price,
                "pnl_pips": pnl_pips,
                "pnl_R": pnl_R,
                "pnl_usd": pnl_usd,
                "mae_R": 0.0,
                "mfe_R": 0.0,
                "fees_usd": trade.commission,
                "tags_json": "{}",
            }
        )
        self.order_meta.pop(trade.ref, None)
