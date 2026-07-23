"""Runtime configuration for Ultralytics in restricted environments."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def configure_ultralytics_directory() -> Path:
    """Use a writable settings directory unless the caller selected one."""
    configured = os.environ.get("YOLO_CONFIG_DIR")
    directory = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "mie1077-ultralytics"
    )
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(directory))
    return directory
