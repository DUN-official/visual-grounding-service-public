"""Console entry points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def serve():
    parser = argparse.ArgumentParser(description="Run the visual grounding HTTP service")
    parser.add_argument("--config", required=True, help="Path to a grounding JSON config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ["GROUNDING_CONFIG"] = str(Path(args.config).expanduser().resolve())
    import uvicorn
    uvicorn.run("grounding.services.api_service:app", host=args.host, port=args.port, reload=args.reload)


def ground_image():
    parser = argparse.ArgumentParser(description="Upload an image and grounding instruction")
    parser.add_argument("image")
    parser.add_argument("instruction")
    parser.add_argument("--target-object")
    parser.add_argument("--location-hint")
    parser.add_argument("--action")
    parser.add_argument("--backend")
    parser.add_argument("--mode", choices=["fast", "balanced", "accurate"], default="balanced")
    parser.add_argument("--maximum-results", type=int, default=None)
    parser.add_argument("--service-url", default="http://127.0.0.1:8000")
    parser.add_argument("--maximum-latency-ms", type=int, default=30_000)
    args = parser.parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    data = {
        "instruction": args.instruction,
        "performance_mode": args.mode,
        "maximum_latency_ms": str(args.maximum_latency_ms),
    }
    if args.maximum_results is not None:
        data["maximum_results"] = str(args.maximum_results)
    for key, value in {
        "target_object": args.target_object,
        "location_hint": args.location_hint,
        "action": args.action,
        "preferred_backend": args.backend,
    }.items():
        if value is not None:
            data[key] = value
    import httpx
    with image_path.open("rb") as handle:
        response = httpx.post(
            f"{args.service_url.rstrip('/')}/v1/ground/upload",
            data=data,
            files={"image": (image_path.name, handle, _media_type(image_path))},
            timeout=max(30.0, args.maximum_latency_ms / 1000.0 + 5.0),
        )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


def _media_type(path):
    return {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
