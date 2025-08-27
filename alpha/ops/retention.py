from __future__ import annotations

"""Retention helpers for pruning old pipeline runs."""

from pathlib import Path
from typing import Dict, Any
import shutil


def cleanup_runs(schedule_id: str, runs_root: str, keep_last: int, delete_artifacts: bool) -> Dict[str, Any]:
    """Remove old run directories keeping only ``keep_last`` most recent."""

    root = Path(runs_root)
    if not root.exists():  # nothing to do
        return {"deleted": [], "kept": [], "bytes_freed": 0}

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    kept = [p.name for p in run_dirs[-keep_last:]] if keep_last else [p.name for p in run_dirs]
    to_delete = run_dirs[:-keep_last] if keep_last else []
    deleted = []
    bytes_freed = 0
    for d in to_delete:
        if delete_artifacts:
            for f in d.rglob("*"):
                if f.is_file():
                    bytes_freed += f.stat().st_size
        shutil.rmtree(d, ignore_errors=True)
        deleted.append(d.name)
    return {"deleted": deleted, "kept": kept, "bytes_freed": bytes_freed}
