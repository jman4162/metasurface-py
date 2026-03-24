"""Reproducibility utilities."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np

import metasurface_py


def capture_environment() -> dict[str, Any]:
    """Capture current environment for reproducibility.

    Returns:
        Dict with package version, Python version, platform,
        git commit hash, and timestamp.
    """
    git_hash = _get_git_hash()
    return {
        "metasurface_py_version": metasurface_py.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": git_hash,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)


def _get_git_hash() -> str:
    """Try to get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"
