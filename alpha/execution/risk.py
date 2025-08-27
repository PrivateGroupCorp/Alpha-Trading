from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict


@dataclass
class RiskCfg:
    """Configuration for position sizing and friction."""

    risk_fixed_pct: float = 0.5
    pip_size: float = 0.0001
    contract_size: int = 100000
    spread_pips: float = 0.1
    slippage_pips: float = 0.1
    commission_per_million: float = 30.0
    sl_pad_atr_mult: float = 0.10


def compute_sl_tp(
    entry_price: float,
    zone_row,
    atr_m1: float,
    direction: str,
    tp_schema: List[Dict[str, float]],
    pip_size: float,
    sl_pad_mult: float,
) -> Dict[str, object]:
    """Compute stop loss and take profit levels.

    Parameters
    ----------
    entry_price: float
        Executed entry price.
    zone_row: pd.Series or mapping
        Must provide ``y_top`` and ``y_bottom`` for zone bounds.
    atr_m1: float
        Current ATR value on M1 timeframe.
    direction: str
        "long" or "short".
    tp_schema: list
        List of dictionaries with keys ``r`` and ``pct`` specifying
        R-multiples for partial exits.
    pip_size: float
        Pip size for instrument.
    sl_pad_mult: float
        Extra padding multiplier for stop beyond the zone.

    Returns
    -------
    dict
        {"sl": float, "tps": [...], "R_per_tp": [...], "risk_per_unit": float}
    """

    top = float(zone_row.get("y_top"))
    bottom = float(zone_row.get("y_bottom"))

    if direction == "long":
        sl = bottom - atr_m1 * sl_pad_mult
        distance = entry_price - sl
        tps = [entry_price + distance * float(tp["r"]) for tp in tp_schema]
    else:
        sl = top + atr_m1 * sl_pad_mult
        distance = sl - entry_price
        tps = [entry_price - distance * float(tp["r"]) for tp in tp_schema]

    distance = max(distance, 1e-12)
    risk_per_unit = distance
    r_per_tp = [float(tp["r"]) for tp in tp_schema]

    return {"sl": sl, "tps": tps, "R_per_tp": r_per_tp, "risk_per_unit": risk_per_unit}


def position_size(equity_usd: float, risk_pct: float, risk_per_unit_usd: float) -> float:
    """Calculate position size in units for fixed percent risk."""

    risk_amount = equity_usd * (risk_pct / 100.0)
    if risk_per_unit_usd <= 0:
        return 0.0
    return risk_amount / risk_per_unit_usd


def fees_and_friction(
    notional_usd: float,
    spread_pips: float,
    slippage_pips: float,
    commission_per_million: float,
    pip_value_per_unit: float,
) -> Dict[str, float]:
    """Estimate trading costs for a single fill."""

    spread_cost = spread_pips * pip_value_per_unit
    slippage_cost = slippage_pips * pip_value_per_unit
    commission = notional_usd * commission_per_million / 1_000_000.0
    return {
        "spread_cost_usd": spread_cost,
        "slippage_cost_usd": slippage_cost,
        "commission_usd": commission,
    }
