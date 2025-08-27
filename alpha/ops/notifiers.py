from __future__ import annotations

"""Simple notification helpers for scheduler results."""

from typing import Any, Dict, List
import json
import smtplib
import urllib.request, urllib.parse
from email.message import EmailMessage


def fmt_summary_card(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a human readable summary card from run result."""

    status_emoji = "✅" if result.get("status") == "success" else "❌"
    kpi = result.get("kpi", {})
    parts: List[str] = [
        f"{status_emoji} Alpha Pipeline — {result.get('symbol')} {result.get('htf')}/{result.get('ltf')}",
        f"trades: {kpi.get('n_trades', 0)} | win_rate: {kpi.get('win_rate', 0)} | avg_R: {kpi.get('avg_R', 0)} | maxDD_R: {kpi.get('maxDD_R', 0)}",
    ]
    if result.get("report_path"):
        parts.append(f"report: {result['report_path']}")
    parts.append(f"run_id: {result.get('run_id')}")
    return {"text": "\n".join(parts)}


def notify_slack(webhook_url: str, card: Dict[str, Any]) -> None:
    data = json.dumps({"text": card["text"]}).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)  # pragma: no cover - side effect


def notify_telegram(bot_token: str, chat_id: str, card: Dict[str, Any]) -> None:
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": card["text"]}).encode("utf-8")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    req = urllib.request.Request(url, data=data)
    urllib.request.urlopen(req)  # pragma: no cover - side effect


def notify_email(smtp_url: str, to: List[str], card: Dict[str, Any]) -> None:
    msg = EmailMessage()
    msg["Subject"] = "Alpha Pipeline Result"
    msg["From"] = smtp_url
    msg["To"] = ",".join(to)
    msg.set_content(card["text"])
    with smtplib.SMTP(smtp_url) as smtp:  # pragma: no cover - external side effect
        smtp.send_message(msg)
