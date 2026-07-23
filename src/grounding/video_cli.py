"""Commands for recorded-video and local-camera grounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .video.processor import VideoGroundingProcessor
from .video.service_adapter import HTTPGroundingServiceAdapter


_BACKENDS = ["auto", "yolo", "owlvit", "gpt_guided_owlvit"]


def _processor(service: HTTPGroundingServiceAdapter) -> VideoGroundingProcessor:
    return VideoGroundingProcessor(
        service,
        tracker_type="AUTO",
        lost_frames=8,
        acquisition_interval_frames=30,
        maximum_acquisition_interval_frames=120,
        smoothing_alpha=0.28,
        jpeg_quality=92,
        tracking_max_width=1280,
        progress_interval_frames=15,
        log_flush_interval_frames=60,
    )


def ground_video() -> None:
    parser = argparse.ArgumentParser(
        description="Ground and track one target in a recorded video"
    )
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("instruction", help="Natural-language target instruction")
    parser.add_argument("--backend", choices=_BACKENDS, default="auto")
    parser.add_argument("--service-url", default="http://127.0.0.1:8000")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--maximum-latency-ms", type=int, default=200_000)
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    service = HTTPGroundingServiceAdapter(args.service_url)
    try:
        service.health()
        summary = _processor(service).process(
            source=str(video_path),
            instruction=args.instruction,
            preferred_backend=None if args.backend == "auto" else args.backend,
            maximum_latency_ms=args.maximum_latency_ms,
            output_video=args.output,
            save_video=True,
            display=args.display,
            use_llm_parser=True,
        )
        print(json.dumps(summary, indent=2))
    finally:
        service.close()


def ground_camera() -> None:
    parser = argparse.ArgumentParser(
        description="Ground and track one target from a local camera"
    )
    parser.add_argument("instruction", help="Natural-language target instruction")
    parser.add_argument("--backend", choices=_BACKENDS, default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--source", default=None, help="RTSP/HTTP source; overrides camera index")
    parser.add_argument("--service-url", default="http://127.0.0.1:8000")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--maximum-latency-ms", type=int, default=200_000)
    args = parser.parse_args()

    source: str | int = args.source if args.source is not None else args.camera_index
    service = HTTPGroundingServiceAdapter(args.service_url)
    try:
        service.health()
        summary = _processor(service).process(
            source=source,
            instruction=args.instruction,
            preferred_backend=None if args.backend == "auto" else args.backend,
            maximum_latency_ms=args.maximum_latency_ms,
            save_video=args.save_video,
            display=True,
            use_llm_parser=True,
        )
        print(json.dumps(summary, indent=2))
    finally:
        service.close()
