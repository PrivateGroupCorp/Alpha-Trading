from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class StageSpec:
    """Specification for a pipeline stage."""

    name: str
    enabled: bool = True
    deps: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    run_fn: Callable[..., Dict] | None = None
    healthcheck_fn: Callable[..., Dict] | None = None
