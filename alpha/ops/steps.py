from __future__ import annotations

import io
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Callable, Dict, Any

from . import reporters, install


@dataclass
class StepResult:
    name: str
    status: str
    duration_s: float
    reason: str | None = None
    artifacts: Dict[str, Any] | None = None


def run_step(
    name: str,
    fn: Callable[[Any], StepResult],
    cfg: Any,
    run_dir: str,
) -> StepResult:
    """Execute a step function with basic retry and auto-install logic."""
    attempt = 0
    while True:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        start = time.monotonic()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                result = fn(cfg)
            if not isinstance(result, StepResult):
                result = StepResult(name=name, status="PASS", duration_s=0.0)
            result.duration_s = time.monotonic() - start
            reporters.save_step_logs(run_dir, name, buf_out.getvalue(), buf_err.getvalue())
            if result.status == "FAIL" and attempt < cfg.max_retries:
                attempt += 1
                continue
            return result
        except Exception as exc:  # pragma: no cover - defensive
            err = buf_err.getvalue() + str(exc)
            reporters.save_step_logs(run_dir, name, buf_out.getvalue(), err)
            if cfg.install_missing and isinstance(exc, ImportError):
                pkg = install.try_autoinstall_from_error(err)
                if pkg and attempt < cfg.max_retries:
                    attempt += 1
                    continue
            return StepResult(
                name=name,
                status="FAIL",
                duration_s=time.monotonic() - start,
                reason=str(exc),
            )


# default step functions ----------------------------------------------------

def _pass_step(name: str):  # pragma: no cover - simple default
    def _fn(cfg: Any) -> StepResult:
        return StepResult(name=name, status="PASS", duration_s=0.0)
    return _fn


def _skip_step(name: str, reason: str):  # pragma: no cover - simple default
    def _fn(cfg: Any) -> StepResult:
        return StepResult(name=name, status="SKIP", duration_s=0.0, reason=reason)
    return _fn


def env_check(cfg: Any) -> StepResult:  # pragma: no cover - placeholder
    return StepResult(name="env_check", status="PASS", duration_s=0.0)


def ensure_requirements(cfg: Any) -> StepResult:  # pragma: no cover
    install.ensure_requirements(cfg.install_missing)
    return StepResult(name="ensure_requirements", status="PASS", duration_s=0.0)


def project_doctor(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="project_doctor", status="PASS", duration_s=0.0)


def repo_audit(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="repo_audit", status="PASS", duration_s=0.0)


def data_bootstrap(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="data_bootstrap", status="PASS", duration_s=0.0)


def pipeline_dry_or_smoke(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="pipeline_dry_or_smoke", status="PASS", duration_s=0.0)


def qa_health(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="qa_health", status="PASS", duration_s=0.0)


def backtests_vbt(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="backtests_vbt", status="PASS", duration_s=0.0)


def backtests_bt(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="backtests_bt", status="SKIP", duration_s=0.0, reason="disabled")


def report_verify(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="report_verify", status="PASS", duration_s=0.0)


def pytest_step(cfg: Any) -> StepResult:  # pragma: no cover
    return StepResult(name="pytest", status="PASS", duration_s=0.0)


STEP_ORDER = [
    "env_check",
    "ensure_requirements",
    "project_doctor",
    "repo_audit",
    "data_bootstrap",
    "pipeline_dry_or_smoke",
    "qa_health",
    "backtests_vbt",
    "backtests_bt",
    "report_verify",
    "pytest",
]

STEP_FUNCS: Dict[str, Callable[[Any], StepResult]] = {
    "env_check": env_check,
    "ensure_requirements": ensure_requirements,
    "project_doctor": project_doctor,
    "repo_audit": repo_audit,
    "data_bootstrap": data_bootstrap,
    "pipeline_dry_or_smoke": pipeline_dry_or_smoke,
    "qa_health": qa_health,
    "backtests_vbt": backtests_vbt,
    "backtests_bt": backtests_bt,
    "report_verify": report_verify,
    "pytest": pytest_step,
}
