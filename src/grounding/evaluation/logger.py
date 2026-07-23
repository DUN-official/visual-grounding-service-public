"""Thread-safe JSONL logger that redacts raw uploaded image data."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class GroundingJSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, request, result) -> None:
        request_data = request.model_dump(mode="json")
        image = request_data.get("image", {})
        encoded = image.get("base64_data")
        if encoded:
            try:
                raw = base64.b64decode(encoded, validate=True)
                image["sha256"] = hashlib.sha256(raw).hexdigest()
                image["size_bytes"] = len(raw)
            except Exception:
                image["sha256"] = None
                image["size_bytes"] = None
            image["base64_data"] = "<redacted>"

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": request_data,
            "result": result.model_dump(mode="json"),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
