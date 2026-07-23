"""Background session manager for recorded-video processing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from .processor import VideoGroundingProcessor
from .service_adapter import HTTPGroundingServiceAdapter


class VideoSessionManager:
    """Run one recorded-video job at a time and expose lightweight status."""

    def __init__(self, output_root: str | Path = "runtime/video_sessions") -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="video-grounding",
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def create(
        self,
        *,
        input_path: Path,
        instruction: str,
        backend: str | None,
        service_url: str,
        use_segmentation: bool = False,
    ) -> str:
        session_id = uuid4().hex[:16]
        session_dir = self.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "status": "queued",
                "phase": "queued",
                "instruction": instruction,
                "backend": backend or "auto",
                "state": "QUEUED",
                "frame_index": 0,
                "total_frames": 0,
                "progress": 0.0,
                "model_calls": 0,
                "video_ready": False,
                "browser_video_compatible": False,
                "use_segmentation": bool(use_segmentation),
                "session_dir": str(session_dir),
            }
        self._executor.submit(
            self._run,
            session_id=session_id,
            input_path=input_path,
            instruction=instruction,
            backend=backend,
            service_url=service_url,
            use_segmentation=use_segmentation,
        )
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return dict(session) if session else None

    def path(self, session_id: str, name: str) -> Path | None:
        session = self.get(session_id)
        if not session:
            return None
        path = Path(session["session_dir"]) / name
        return path if path.exists() and path.stat().st_size > 0 else None

    def _run(
        self,
        *,
        session_id: str,
        input_path: Path,
        instruction: str,
        backend: str | None,
        service_url: str,
        use_segmentation: bool = False,
    ) -> None:
        self._update(session_id, status="processing", phase="parsing", state="PARSING")
        service = HTTPGroundingServiceAdapter(service_url)
        try:
            processor = VideoGroundingProcessor(
                service,
                output_root=self.output_root,
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
            summary = processor.process(
                source=str(input_path),
                instruction=instruction,
                preferred_backend=backend,
                maximum_latency_ms=200_000,
                save_video=True,
                display=False,
                session_name=session_id,
                progress_callback=lambda data, frame: self._progress(session_id, data),
                use_llm_parser=True,
                use_segmentation=use_segmentation,
            )

            summary_update = dict(summary)
            summary_update.pop("session_id", None)

            # Normalize the final session state for the browser.  The processor
            # reports ``final_state`` and ``frames_processed`` while the session
            # status endpoint exposes ``state`` and ``frame_index``.
            summary_update["status"] = "completed"
            summary_update["phase"] = "completed"
            summary_update["state"] = (
                summary_update.get("final_state") or "COMPLETED"
            )
            summary_update["progress"] = 1.0
            summary_update["frame_index"] = summary_update.get(
                "frames_processed",
                summary_update.get("frame_index", 0),
            )
            summary_update["total_frames"] = summary_update.get(
                "total_source_frames",
                summary_update.get("total_frames", 0),
            )
            summary_update["error"] = None

            self._update(session_id, **summary_update)
        except Exception as exc:
            self._update(
                session_id,
                status="failed",
                phase="failed",
                state="FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            service.close()

    def _progress(self, session_id: str, data: dict[str, Any]) -> None:
        allowed = {
            "status",
            "phase",
            "frame_index",
            "total_frames",
            "progress",
            "state",
            "backend",
            "tracker",
            "tracker_quality",
            "model_calls",
            "successful_acquisitions",
            "last_grounding_status",
            "last_grounding_error",
            "plan",
            "video_ready",
            "browser_video_compatible",
            "video_encoding_error",
            "use_segmentation",
        }
        self._update(
            session_id,
            **{key: value for key, value in data.items() if key in allowed},
        )

    def _update(self, session_id: str, **values: Any) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].update(values)
