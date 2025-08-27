from __future__ import annotations

import platform
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from . import steps, reporters


@dataclass
class RunAllCfg:
    profile: str
    symbol: str
    htf: str
    ltf: str
    mode: str
    rolling_days: int
    install_missing: bool
    auto_fetch: bool
    provider: str
    continue_on_error: bool
    max_retries: int
    timeout_per_step_sec: int
    run_bt: bool
    skip_backtests_vbt: bool


def _grade(steps_data: List[Dict[str, Any]]) -> str:
    non_py_fail = any(
        st["status"] == "FAIL" and st["name"] != "pytest" for st in steps_data
    )
    if non_py_fail:
        return "RED"
    py_fail = any(st["name"] == "pytest" and st["status"] == "FAIL" for st in steps_data)
    if py_fail:
        return "YELLOW"
    return "GREEN"


def run_all(cfg: RunAllCfg) -> Dict[str, Any]:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts/run-all") / f"{cfg.symbol}_{cfg.htf}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "inputs": {
            "symbol": cfg.symbol,
            "htf": cfg.htf,
            "ltf": cfg.ltf,
            "mode": cfg.mode,
        },
        "env": {"python": platform.python_version()},
        "steps": [],
    }

    for name in steps.STEP_ORDER:
        fn = steps.STEP_FUNCS[name]
        result = steps.run_step(name, fn, cfg, str(run_dir))
        summary["steps"].append(asdict(result))
        if result.status == "FAIL" and not cfg.continue_on_error and name != "pytest":
            break

    summary["grade"] = _grade(summary["steps"])
    summary["exit_code"] = 0 if summary["grade"] in {"GREEN", "YELLOW"} else 1

    summary_json = reporters.write_summary_json(str(run_dir), summary)
    summary_md = reporters.write_summary_md(str(run_dir), summary)
    summary["summary_json"] = summary_json
    summary["summary_md"] = summary_md
    summary["run_dir"] = str(run_dir)
    return summary
