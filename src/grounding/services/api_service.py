"""FastAPI service for image, recorded-video, and best-effort live-camera grounding."""

from __future__ import annotations

from contextlib import asynccontextmanager
import base64
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from ..config import load_config
from ..llm_task_parser import parse_task_with_service
from ..video.service_adapter import build_video_grounding_plan
from ..schemas import (
    GroundingRequest,
    GroundingResult,
    ImagePayload,
    PerformanceMode,
    QuantityIntent,
)
from ..video.live_session import LiveCameraSessionManager
from ..video.session_manager import VideoSessionManager
from .local_service import LocalGroundingService

ALLOWED_UPLOAD_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_VIDEO_BACKENDS = {"auto", "yolo", "owlvit", "gpt_guided_owlvit"}
VIDEO_MAX_BYTES = 1024 * 1024 * 1024


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def create_app(config_path: str | Path | None = None):
    root = repository_root()
    chosen = Path(
        config_path
        or os.environ.get(
            "GROUNDING_CONFIG",
            root / "configs" / "grounding_service.json",
        )
    )
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(app):
        config = load_config(chosen)
        service = LocalGroundingService.from_config(config)
        service.startup()
        state["service"] = service
        state["max_image_bytes"] = config.service.max_image_bytes
        state["video_manager"] = VideoSessionManager(
            root / "runtime" / "video_sessions"
        )
        state["live_manager"] = LiveCameraSessionManager()
        yield
        video_manager = state.get("video_manager")
        if isinstance(video_manager, VideoSessionManager):
            video_manager.shutdown()
        live_manager = state.get("live_manager")
        if isinstance(live_manager, LiveCameraSessionManager):
            live_manager.shutdown()
        service.shutdown()
        state.clear()

    app = FastAPI(
        title="MIE1077 Adaptive Visual Grounding Service",
        version="0.6.0",
        lifespan=lifespan,
    )

    def get_service() -> LocalGroundingService:
        service = state.get("service")
        if not isinstance(service, LocalGroundingService):
            raise HTTPException(status_code=503, detail="service not ready")
        return service

    def get_video_manager() -> VideoSessionManager:
        manager = state.get("video_manager")
        if not isinstance(manager, VideoSessionManager):
            raise HTTPException(status_code=503, detail="video service not ready")
        return manager

    def get_live_manager() -> LiveCameraSessionManager:
        manager = state.get("live_manager")
        if not isinstance(manager, LiveCameraSessionManager):
            raise HTTPException(status_code=503, detail="live camera service not ready")
        return manager

    @app.get("/", response_class=HTMLResponse)
    def browser_interface():
        html = (root / "src" / "grounding" / "web" / "index.html").read_text(
            encoding="utf-8"
        )
        return HTMLResponse(html)

    @app.get("/health")
    def health():
        service = get_service()
        return {
            "service": "ready",
            "backends": {
                name: item.model_dump(mode="json")
                for name, item in service.health().items()
            },
            "startup_errors": service.startup_errors,
            "recorded_video": "ready",
            "live_camera": "best_effort",
        }

    @app.post("/v1/route")
    def route(request: GroundingRequest):
        service = get_service()
        effective, parser_trace = service.prepare_request(request)
        decision = service.router.route(effective, service.health())
        return {
            "parsed_request": effective.model_dump(mode="json"),
            "parser_trace": parser_trace.model_dump(mode="json"),
            "routing": decision.model_dump(mode="json"),
        }

    @app.post("/v1/task/parse")
    def parse_video_task(payload: dict[str, object]):
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise HTTPException(status_code=400, detail="instruction is required")
        performance_mode = str(payload.get("performance_mode") or "balanced")
        parser_mode = str(payload.get("parser_mode") or "llm")
        preferred_backend = payload.get("preferred_backend")
        structured = parse_task_with_service(
            get_service(),
            instruction,
            parser_mode=parser_mode,
        )
        plan = build_video_grounding_plan(
            instruction,
            performance_mode=performance_mode,
            preferred_backend=(
                str(preferred_backend) if preferred_backend is not None else None
            ),
            parser_mode=parser_mode,
            structured=structured,
        )
        return {
            "instruction": instruction,
            "target_object": plan.target_object,
            "target_phrase": plan.target_phrase,
            "attributes": plan.attributes,
            "relations": plan.relations,
            "anchor_objects": plan.anchor_objects,
            "anchor_phrases": plan.anchor_phrases,
            "parser_source": structured.parser_source,
            "parser_confidence": structured.parser_confidence,
            "fallback_reason": structured.fallback_reason,
            "requested_backend": plan.requested_backend,
            "grounding_instruction": plan.grounding_instruction,
        }
    @app.post("/v1/ground", response_model=GroundingResult)
    def ground(request: GroundingRequest):
        return get_service().ground(request)

    @app.post("/v1/ground/upload", response_model=GroundingResult)
    async def ground_upload(
        image: UploadFile = File(...),
        instruction: str = Form(...),
        target_object: str | None = Form(default=None),
        location_hint: str | None = Form(default=None),
        action: str | None = Form(default=None),
        performance_mode: PerformanceMode = Form(default=PerformanceMode.BALANCED),
        maximum_results: int | None = Form(default=None),
        maximum_latency_ms: int = Form(default=200_000),
        preferred_backend: str | None = Form(default=None),
        return_segmentation: bool = Form(default=False),
    ):
        media_type = (image.content_type or "").lower()
        if media_type not in ALLOWED_UPLOAD_MEDIA_TYPES:
            raise HTTPException(
                status_code=415,
                detail="unsupported image type; use JPEG, PNG, or WebP",
            )
        maximum_bytes = int(state.get("max_image_bytes", 30 * 1024 * 1024))
        image_bytes = await image.read(maximum_bytes + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="uploaded image is empty")
        if len(image_bytes) > maximum_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"image exceeds {maximum_bytes} byte limit",
            )
        request_data: dict[str, object] = {
            "image": ImagePayload(
                base64_data=base64.b64encode(image_bytes).decode("ascii"),
                media_type=media_type,
            ),
            "instruction": instruction,
            "target_object": target_object,
            "location_hint": location_hint,
            "action": action,
            "performance_mode": performance_mode,
            "maximum_latency_ms": maximum_latency_ms,
            "preferred_backend": preferred_backend,
            "metadata": {
                "uploaded_filename": image.filename,
                "input_mode": "multipart_upload",
                "return_segmentation": bool(return_segmentation),
            },
        }
        if maximum_results is not None:
            request_data["maximum_results"] = maximum_results
            # Selecting more than one result is an explicit request for
            # multi-instance image grounding, even when the written prompt is
            # singular. Video tracking remains single-target.
            if maximum_results > 1:
                request_data["quantity"] = QuantityIntent.MULTIPLE
        request_model = GroundingRequest(**request_data)
        return await run_in_threadpool(get_service().ground, request_model)

    @app.post("/v1/video/upload")
    async def video_upload(
        request: Request,
        video: UploadFile = File(...),
        instruction: str = Form(...),
        backend: str = Form(default="auto"),
        use_segmentation: bool = Form(default=False),
    ):
        suffix = Path(video.filename or "video.mp4").suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise HTTPException(status_code=415, detail="unsupported video type")
        backend = backend.strip().lower()
        if backend not in ALLOWED_VIDEO_BACKENDS:
            raise HTTPException(status_code=400, detail="unsupported backend")
        if not instruction.strip():
            raise HTTPException(status_code=400, detail="instruction is required")

        upload_root = root / "runtime" / "video_uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        upload_path = upload_root / f"{uuid4().hex}{suffix}"
        size = 0
        with upload_path.open("wb") as handle:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > VIDEO_MAX_BYTES:
                    handle.close()
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="video exceeds 1 GiB limit",
                    )
                handle.write(chunk)
        if size == 0:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="uploaded video is empty")

        session_id = get_video_manager().create(
            input_path=upload_path,
            instruction=instruction.strip(),
            backend=None if backend == "auto" else backend,
            service_url=str(request.base_url).rstrip("/"),
            use_segmentation=bool(use_segmentation),
        )
        return {
            "session_id": session_id,
            "status_url": f"/v1/video/sessions/{session_id}",
        }

    @app.get("/v1/video/sessions/{session_id}")
    def video_status(session_id: str):
        manager = get_video_manager()
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="video session not found")
        video_path = manager.path(session_id, "annotated_video.mp4")
        session["video_ready"] = bool(
            session.get("video_ready") and video_path is not None
        )
        session["video_url"] = f"/v1/video/sessions/{session_id}/video"
        session["summary_url"] = f"/v1/video/sessions/{session_id}/summary"
        session["predictions_url"] = (
            f"/v1/video/sessions/{session_id}/predictions"
        )
        return session

    @app.get("/v1/video/sessions/{session_id}/video")
    def video_result(session_id: str):
        path = get_video_manager().path(session_id, "annotated_video.mp4")
        if path is None:
            raise HTTPException(status_code=404, detail="annotated video not ready")
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'inline; filename="annotated_video.mp4"',
            },
        )

    @app.get("/v1/video/sessions/{session_id}/summary")
    def video_summary(session_id: str):
        path = get_video_manager().path(session_id, "summary.json")
        if path is None:
            raise HTTPException(status_code=404, detail="summary not ready")
        return FileResponse(
            path,
            media_type="application/json",
            filename="summary.json",
        )

    @app.get("/v1/video/sessions/{session_id}/predictions")
    def video_predictions(session_id: str):
        path = get_video_manager().path(session_id, "predictions.jsonl")
        if path is None:
            raise HTTPException(status_code=404, detail="predictions not ready")
        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename="predictions.jsonl",
        )

    @app.post("/v1/live/start")
    async def live_start(
        request: Request,
        instruction: str = Form(...),
        backend: str = Form(default="auto"),
        use_segmentation: bool = Form(default=False),
    ):
        backend = backend.strip().lower()
        if backend not in ALLOWED_VIDEO_BACKENDS:
            raise HTTPException(status_code=400, detail="unsupported backend")
        if not instruction.strip():
            raise HTTPException(status_code=400, detail="instruction is required")
        session_id = get_live_manager().create(
            instruction=instruction.strip(),
            backend=None if backend == "auto" else backend,
            camera_index=0,
            service_url=str(request.base_url).rstrip("/"),
            use_segmentation=bool(use_segmentation),
        )
        return {
            "session_id": session_id,
            "status_url": f"/v1/live/sessions/{session_id}",
            "stream_url": f"/v1/live/sessions/{session_id}/stream",
        }

    @app.get("/v1/live/sessions/{session_id}")
    def live_status(session_id: str):
        session = get_live_manager().get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="live session not found")
        session["stream_url"] = f"/v1/live/sessions/{session_id}/stream"
        return session

    @app.get("/v1/live/sessions/{session_id}/stream")
    def live_stream(session_id: str):
        if get_live_manager().get(session_id) is None:
            raise HTTPException(status_code=404, detail="live session not found")
        return StreamingResponse(
            get_live_manager().mjpeg_stream(session_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/live/sessions/{session_id}/reset")
    def live_reset(session_id: str):
        if not get_live_manager().reset_tracking(session_id):
            raise HTTPException(
                status_code=404,
                detail="live session not found or no longer active",
            )
        return {
            "session_id": session_id,
            "status": "reacquiring",
            "message": "Tracker reset; acquiring a fresh target on the live feed",
        }

    @app.post("/v1/live/sessions/{session_id}/stop")
    def live_stop(session_id: str):
        if not get_live_manager().stop(session_id):
            raise HTTPException(status_code=404, detail="live session not found")
        return {"session_id": session_id, "status": "stopping"}

    return app


app = create_app()
