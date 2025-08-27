from __future__ import annotations
from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass

import backtrader as bt


@dataclass
class BrokerCfg:
    spread_pips: float = 0.1
    slippage_pips: float = 0.1
    commission_per_million: float = 30.0
    comm_apply_both_sides: bool = True
    pip_size: float = 0.0001
    contract_size: float = 100000


class _SpreadCommissionInfo(bt.CommInfoBase):
    """Commission scheme applying spread, slippage and per notional fees."""

    def __init__(self, cfg: BrokerCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def getoperationprice(self, price, isbuy):  # type: ignore[override]
        adj = (self.cfg.spread_pips / 2 + self.cfg.slippage_pips) * self.cfg.pip_size
        return price + adj if isbuy else price - adj

    def getcommission(self, size, price, pseudoexec=False):  # type: ignore[override]
        notional = abs(size) * price
        comm = notional * self.cfg.commission_per_million / 1_000_000
        if self.cfg.comm_apply_both_sides:
            comm /= 2.0
        return comm


def build_broker_and_sizers(cerebro: bt.Cerebro, cfg: BrokerCfg) -> None:
    """Configure broker with commission, spread and slippage models."""
    comminfo = _SpreadCommissionInfo(cfg)
    cerebro.broker.addcommissioninfo(comminfo)
