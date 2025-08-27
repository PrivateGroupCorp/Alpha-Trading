import json
from pathlib import Path
from typing import Any
import os

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    """Safely load a JSON file and return a dict."""
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {}


def load_csv(path: Path) -> pd.DataFrame:
    """Safely load a CSV file and return a DataFrame."""
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def format_float(value: Any, decimals: int) -> str:
    """Format a value as float with given decimals."""
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return ""


def relpath(path: Path, start: Path) -> str:
    """Return a POSIX relative path."""
    return os.path.relpath(path, start).replace(os.sep, "/")
