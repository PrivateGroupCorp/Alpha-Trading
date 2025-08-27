from __future__ import annotations

"""Scheduler for running pipeline profiles periodically."""

from dataclasses import dataclass
from typing import Dict, Any
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yaml

from alpha.ops.runner import RunCfg, run_pipeline

from .utils import build_window, next_cron_time
from .retention import cleanup_runs
from .notifiers import fmt_summary_card, notify_slack, notify_telegram, notify_email


@dataclass
class ScheduleCfg:
    enable: bool
    profile: str
    symbol: str
    htf: str
    ltf: str
    window: Dict[str, Any]
    when: Dict[str, Any]
    behavior: Dict[str, Any]
    backoff: Dict[str, Any]
    notify: Dict[str, Any]
    retention: Dict[str, Any]


def _load_profile(profile: str) -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "pipeline.yml"
    data = yaml.safe_load(cfg_path.read_text())
    return data["profiles"][profile]


def run_once(schedule_id: str, cfg: ScheduleCfg) -> Dict[str, Any]:
    """Execute pipeline for a schedule once and record results."""

    logger = logging.getLogger(f"alpha.ops.scheduler.{schedule_id}")
    window = build_window(cfg.window)

    prof = _load_profile(cfg.profile)
    behavior = prof.get("behavior", {}).copy()
    behavior.update(cfg.behavior or {})
    io = prof.get("io", {}).copy()
    art_root_override = os.getenv("ALPHA_ARTIFACTS_ROOT")
    if art_root_override:
        io["artifacts_root"] = art_root_override
        io["runs_root"] = str(Path(art_root_override) / "runs")
    run_cfg = RunCfg(
        profile=cfg.profile,
        symbol=cfg.symbol,
        htf=cfg.htf,
        ltf=cfg.ltf,
        start=window.start,
        end=window.end,
        stages=prof["stages"],
        behavior=behavior,
        io=io,
    )

    retries = int(cfg.backoff.get("retries", 0))
    delay = float(cfg.backoff.get("initial_s", 0))
    factor = float(cfg.backoff.get("factor", 2.0))
    max_s = float(cfg.backoff.get("max_s", delay))
    attempt = 0
    while True:
        try:
            result = run_pipeline(run_cfg)
            break
        except Exception:  # pragma: no cover - defensive
            attempt += 1
            if attempt > retries:
                raise
            wait_s = min(max_s, delay * (factor ** (attempt - 1)))
            time.sleep(wait_s)

    manifest_path = Path(result["manifest_path"])
    run_dir = manifest_path.parent
    run_id = run_dir.name

    kpi: Dict[str, Any] = {}
    ts_path = run_dir / "trades_summary.json"
    if ts_path.exists():
        kpi.update(json.loads(ts_path.read_text()))
    bt_path = run_dir / "bt_summary.json"
    if bt_path.exists():
        kpi.update(json.loads(bt_path.read_text()))

    reports_dir = Path(io["artifacts_root"]) / "reports" / f"{cfg.symbol}_{cfg.htf}"
    report_path = ""
    if reports_dir.exists():
        htmls = sorted(reports_dir.glob("report*.html"))
        if htmls:
            report_path = str(htmls[-1])

    status = "success" if result.get("status") == "ok" else "failure"
    sched_root = Path(io["artifacts_root"]) / "scheduler" / schedule_id
    sched_root.mkdir(parents=True, exist_ok=True)
    record = {
        "schedule_id": schedule_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "symbol": cfg.symbol,
        "htf": cfg.htf,
        "ltf": cfg.ltf,
        "window": {"start": window.start, "end": window.end},
        "kpi": kpi,
        "report_path": report_path,
        "run_id": run_id,
        "artifacts_root": str(run_dir),
        "scheduler_root": str(sched_root),
    }

    with (sched_root / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    with (sched_root / "last_status.json").open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    policy = cfg.notify.get("on", []) if cfg.notify else []
    if "always" in policy or status in policy:
        card = fmt_summary_card(record)
        for name, ch in (cfg.notify.get("channels") or {}).items():
            if not ch.get("enabled"):
                continue
            try:
                if name == "slack":
                    url = os.getenv(ch.get("webhook_env", ""))
                    if not url:
                        logger.warning("missing env %s for slack", ch.get("webhook_env"))
                        continue
                    notify_slack(url, card)
                elif name == "telegram":
                    token = os.getenv(ch.get("bot_token_env", ""))
                    chat_id = os.getenv(ch.get("chat_id_env", ""))
                    if not token or not chat_id:
                        logger.warning("missing env for telegram")
                        continue
                    notify_telegram(token, chat_id, card)
                elif name == "email":
                    smtp = os.getenv(ch.get("smtp_env", ""))
                    if not smtp:
                        logger.warning("missing env %s for email", ch.get("smtp_env"))
                        continue
                    to = ch.get("to", [])
                    notify_email(smtp, to, card)
            except Exception as exc:  # pragma: no cover - external failures
                logger.warning("notify %s failed: %s", name, exc)

    keep = int(cfg.retention.get("runs_to_keep", 0)) if cfg.retention else 0
    if keep:
        cleanup_runs(schedule_id, str(run_dir.parent), keep, bool(cfg.retention.get("delete_old_artifacts", False)))

    return record


def start_scheduler(all_cfg: Dict[str, ScheduleCfg]) -> None:
    logger = logging.getLogger("alpha.ops.scheduler")
    scheds: Dict[str, Dict[str, Any]] = {}
    for sid, cfg in all_cfg.items():
        if not cfg.enable:
            continue
        tz = cfg.when.get("tz", "UTC")
        cron = cfg.when.get("cron", "* * * * *")
        next_time = next_cron_time(cron, tz, datetime.now(ZoneInfo(tz)) - timedelta(minutes=1))
        scheds[sid] = {"cfg": cfg, "tz": tz, "cron": cron, "next": next_time}

    try:
        while True:
            for sid, info in scheds.items():
                now = datetime.now(ZoneInfo(info["tz"]))
                if now >= info["next"]:
                    run_once(sid, info["cfg"])
                    info["next"] = next_cron_time(info["cron"], info["tz"], info["next"])
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("scheduler stopped")
        return
