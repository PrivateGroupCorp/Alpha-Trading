from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def ensure_requirements(install_missing: bool) -> Dict[str, Any]:
    """Install project requirements if requested.

    Parameters
    ----------
    install_missing: bool
        Whether to attempt installation using ``requirements.txt``.

    Returns
    -------
    Dict[str, Any]
        A result dictionary containing the return code of the pip
        invocation.  Any installation output is captured by the caller.
    """
    req_file = Path("requirements.txt")
    if not install_missing or not req_file.exists():
        return {"ok": True, "returncode": 0}

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    freeze_path = req_file.parent / "pip_freeze.txt"
    with freeze_path.open("w", encoding="utf-8") as fh:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], stdout=fh, text=True
        )

    return {"ok": proc.returncode == 0, "returncode": proc.returncode}


def try_autoinstall_from_error(stderr: str) -> Optional[str]:
    """Attempt to ``pip install`` a package based on an ImportError message.

    Parameters
    ----------
    stderr: str
        The stderr text from a failed step.  If it contains a pattern like
        ``No module named 'pkg'`` the function will attempt to install
        ``pkg`` using pip and return the package name.

    Returns
    -------
    Optional[str]
        The name of the package that was attempted to install, or ``None``
        if no installation was triggered.
    """
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
    if not match:
        return None
    pkg = match.group(1)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg],
        capture_output=True,
        text=True,
    )
    return pkg
