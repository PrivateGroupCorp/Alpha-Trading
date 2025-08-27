from __future__ import annotations

"""Utilities for scheduler: window calculation and cron helpers."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

try:  # pragma: no cover - optional dependency
    from croniter import croniter  # type: ignore
except Exception:  # pragma: no cover - fallback when croniter missing
    croniter = None  # type: ignore


@dataclass
class TimeWindow:
    start: str
    end: str


def build_window(cfg: dict) -> TimeWindow:
    """Build start/end date window from scheduler config.

    If mode is ``rolling_days`` the window ends at today's UTC date and
    starts ``days`` back. For ``fixed`` the start/end strings are used as-is.
    """

    mode = cfg.get("mode", "rolling_days")
    if mode == "rolling_days":
        days = int(cfg.get("days", 0))
        end_dt = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=days)
        return TimeWindow(start=start_dt.isoformat(), end=end_dt.isoformat())
    if mode == "fixed":
        return TimeWindow(start=str(cfg.get("start")), end=str(cfg.get("end")))
    raise ValueError(f"unknown window mode: {mode}")


def next_cron_time(cron: str, tz: str, base: datetime | None = None) -> datetime:
    """Return next run time for ``cron`` in timezone ``tz``.

    Uses ``croniter`` when available; otherwise a very small fallback
    supporting only ``* * * * *`` (every minute).
    """

    tzinfo = ZoneInfo(tz)
    now = base or datetime.now(tzinfo)
    if croniter is None:
        # Fallback: assume every minute
        return (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    return croniter(cron, now).get_next(datetime)  # type: ignore[arg-type]
