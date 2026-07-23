"""Run repeatable latency benchmarks against a running grounding service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a running grounding service")
    parser.add_argument("manifest", help="JSON list containing image and instruction fields")
    parser.add_argument("--service-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="runtime/benchmarks/latency_report.json")
    parser.add_argument("--mode", choices=["fast", "balanced", "accurate"], default="balanced")
    parser.add_argument("--maximum-latency-ms", type=int, default=30_000)
    args = parser.parse_args()

    import httpx

    manifest_path = Path(args.manifest).expanduser().resolve()
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("benchmark manifest must be a non-empty JSON list")

    records = []
    for index, item in enumerate(items, start=1):
        image_path = Path(item["image"]).expanduser()
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        instruction = str(item["instruction"])
        mode = item.get("performance_mode", args.mode)
        backend = item.get("preferred_backend")
        maximum_results = item.get("maximum_results")
        data = {
            "instruction": instruction,
            "performance_mode": mode,
            "maximum_latency_ms": str(args.maximum_latency_ms),
        }
        if backend:
            data["preferred_backend"] = backend
        if maximum_results is not None:
            data["maximum_results"] = str(maximum_results)

        with image_path.open("rb") as handle:
            response = httpx.post(
                f"{args.service_url.rstrip('/')}/v1/ground/upload",
                data=data,
                files={"image": (image_path.name, handle, _media_type(image_path))},
                timeout=max(30.0, args.maximum_latency_ms / 1000.0 + 5.0),
            )
        payload = response.json()
        record = {
            "index": index,
            "image": str(image_path),
            "instruction": instruction,
            "status_code": response.status_code,
            "status": payload.get("status"),
            "backend_used": payload.get("backend_used"),
            "latency_ms": payload.get("latency_ms"),
            "pipeline_path": payload.get("metadata", {}).get("pipeline_path", []),
            "gpt_request_count": payload.get("metadata", {}).get("gpt_request_count", 0),
            "stage_latencies_ms": payload.get("metadata", {}).get("stage_latencies_ms", {}),
            "error": payload.get("error"),
        }
        records.append(record)
        print(f"[{index}/{len(items)}] {record['status']} {record['backend_used']} {record['latency_ms']} ms")

    successful = [float(r["latency_ms"]) for r in records if r.get("status") == "success" and r.get("latency_ms") is not None]
    summary = {
        "request_count": len(records),
        "success_count": len(successful),
        "mean_latency_ms": mean(successful) if successful else None,
        "median_latency_ms": median(successful) if successful else None,
        "minimum_latency_ms": min(successful) if successful else None,
        "maximum_latency_ms": max(successful) if successful else None,
        "records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved benchmark report: {output}")


def _media_type(path: Path) -> str:
    return {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")


if __name__ == "__main__":
    main()
