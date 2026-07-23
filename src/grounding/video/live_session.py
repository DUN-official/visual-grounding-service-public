"""Best-effort asynchronous live-camera grounding and tracking sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

from ..llm_task_parser import LLMVideoTaskParser, VideoTaskPlan
from ..task_parser import normalize_text
from ..segmentation import draw_segmentation_overlay, segment_from_box
from .service_adapter import HTTPGroundingServiceAdapter
from .tracker import OpenCVBoxTracker


Box = tuple[float, float, float, float]


@dataclass
class _LiveSession:
    session_id: str
    instruction: str
    backend: str | None
    camera_index: int
    service_url: str
    use_segmentation: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    reset_event: threading.Event = field(default_factory=threading.Event)
    reset_generation: int = 0
    status: dict[str, Any] = field(default_factory=dict)
    latest_jpeg: bytes | None = None
    frame_version: int = 0
    thread: threading.Thread | None = None


class LiveCameraSessionManager:
    """Keep camera capture responsive while semantic acquisition runs remotely.

    The expensive grounding call runs only for initial acquisition or sustained
    tracking loss. Camera capture continues in a separate thread. GPT-guided
    acquisition can still take several seconds; once acquired, local tracking is
    near-live on suitable hardware.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _LiveSession] = {}
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop_event.set()
        for session in sessions:
            if session.thread and session.thread.is_alive():
                session.thread.join(timeout=2.0)

    def create(
        self,
        *,
        instruction: str,
        backend: str | None,
        camera_index: int,
        service_url: str,
        use_segmentation: bool = False,
    ) -> str:
        session_id = uuid4().hex[:16]
        session = _LiveSession(
            session_id=session_id,
            instruction=instruction,
            backend=backend,
            camera_index=int(camera_index),
            service_url=service_url,
            use_segmentation=bool(use_segmentation),
            status={
                "session_id": session_id,
                "status": "starting",
                "state": "STARTING",
                "instruction": instruction,
                "backend": backend or "auto",
                "model_calls": 0,
                "reset_count": 0,
                "tracking_fps": 0.0,
                "message": "Opening camera",
                "use_segmentation": bool(use_segmentation),
            },
        )
        thread = threading.Thread(
            target=self._run,
            args=(session,),
            daemon=True,
            name=f"live-camera-{session_id}",
        )
        session.thread = thread
        with self._lock:
            self._sessions[session_id] = session
        thread.start()
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        session = self._session(session_id)
        if session is None:
            return None
        with session.lock:
            return dict(session.status)

    def stop(self, session_id: str) -> bool:
        session = self._session(session_id)
        if session is None:
            return False
        session.stop_event.set()
        return True

    def reset_tracking(self, session_id: str) -> bool:
        """Clear only the selected target and reacquire on the newest frame.

        The camera, MJPEG stream, instruction, parsed plan, and selected backend
        remain active. A generation counter prevents a slow grounding result
        started before the reset from re-locking the stale target afterward.
        """

        session = self._session(session_id)
        if session is None:
            return False
        with session.lock:
            if str(session.status.get("status", "")) in {"stopped", "failed"}:
                return False
            session.reset_generation += 1
            reset_count = int(session.status.get("reset_count", 0)) + 1
            session.status.update(
                status="running",
                state="REACQUIRING",
                tracker=None,
                reset_count=reset_count,
                message="Reset requested; acquiring a fresh target",
            )
        session.reset_event.set()
        return True

    def mjpeg_stream(self, session_id: str) -> Iterator[bytes]:
        session = self._session(session_id)
        if session is None:
            return
        last_version = -1
        while True:
            with session.lock:
                jpeg = session.latest_jpeg
                version = session.frame_version
                status = str(session.status.get("status", ""))
            if jpeg is not None and version != last_version:
                last_version = version
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            if status in {"stopped", "failed"} and version == last_version:
                break
            time.sleep(0.03)

    def _session(self, session_id: str) -> _LiveSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("live camera support requires OpenCV") from exc
        return cv2

    def _run(self, session: _LiveSession) -> None:
        cv2 = self._cv2()
        capture = cv2.VideoCapture(session.camera_index)
        if not capture.isOpened():
            self._update(
                session,
                status="failed",
                state="FAILED",
                message=f"Could not open camera index {session.camera_index}",
            )
            return
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        frame_lock = threading.Lock()
        latest: dict[str, Any] = {"frame": None, "sequence": -1}
        capture_done = threading.Event()

        def capture_loop() -> None:
            sequence = 0
            while not session.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                with frame_lock:
                    latest["frame"] = frame
                    latest["sequence"] = sequence

                # Keep the browser camera view alive while a slow semantic
                # acquisition call is running in the processing thread.
                if sequence % 3 == 0:
                    with session.lock:
                        session_state = str(session.status.get("state", ""))
                    if session_state in {
                        "STARTING", "PARSING", "ACQUIRING", "REACQUIRING"
                    }:
                        preview = self._preview_frame(frame.copy(), 960)
                        cv2.putText(
                            preview,
                            f"{session_state}: keep target visible",
                            (18, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        encoded_ok, encoded = cv2.imencode(
                            ".jpg",
                            preview,
                            [cv2.IMWRITE_JPEG_QUALITY, 78],
                        )
                        if encoded_ok:
                            with session.lock:
                                session.latest_jpeg = encoded.tobytes()
                                session.frame_version += 1
                sequence += 1
            capture_done.set()

        capture_thread = threading.Thread(
            target=capture_loop,
            daemon=True,
            name=f"capture-{session.session_id}",
        )
        capture_thread.start()

        service = HTTPGroundingServiceAdapter(session.service_url)
        parser = LLMVideoTaskParser()
        tracker = OpenCVBoxTracker("AUTO")
        current_box: Box | None = None
        misses = 0
        model_calls = 0
        failure_streak = 0
        next_attempt_time = 0.0
        last_sequence = -1
        processed_since_rate = 0
        rate_started = time.perf_counter()

        try:
            self._update(
                session,
                status="running",
                state="PARSING",
                message="Parsing instruction once",
            )
            plan = parser.parse(session.instruction, use_llm=True)
            selected_backend = self._select_backend(session.backend, plan)
            self._update(
                session,
                state="ACQUIRING",
                backend=selected_backend or "auto",
                parser_source=plan.parser_source,
                target_object=plan.target_object,
                message="Acquiring target; keep it briefly visible and reasonably still",
            )

            while not session.stop_event.is_set():
                if session.reset_event.is_set():
                    tracker.reset()
                    current_box = None
                    misses = 0
                    failure_streak = 0
                    next_attempt_time = 0.0
                    session.reset_event.clear()
                    self._update(
                        session,
                        status="running",
                        state="REACQUIRING",
                        tracker=None,
                        message="Tracker cleared; reacquiring on the newest frame",
                    )

                with frame_lock:
                    sequence = int(latest["sequence"])
                    raw_frame = latest["frame"]
                    frame = None if raw_frame is None else raw_frame.copy()
                if frame is None or sequence == last_sequence:
                    time.sleep(0.01)
                    continue
                last_sequence = sequence
                working, scale = self._tracking_frame(frame, 1280)
                state = "TRACKING" if current_box is not None else "ACQUIRING"

                if current_box is None and time.monotonic() >= next_attempt_time:
                    self._update(
                        session,
                        state="ACQUIRING",
                        message="Semantic acquisition in progress; camera capture remains active",
                    )
                    with session.lock:
                        acquisition_generation = session.reset_generation
                    observation = service.ground_frame(
                        frame,
                        instruction=plan.instruction,
                        target_object=plan.target_object,
                        location_hint=plan.location_hint,
                        action="find",
                        performance_mode="balanced",
                        preferred_backend=selected_backend,
                        maximum_latency_ms=200_000,
                        jpeg_quality=92,
                    )
                    model_calls += 1
                    with session.lock:
                        stale_after_reset = (
                            acquisition_generation != session.reset_generation
                        )
                    if stale_after_reset:
                        # A reset was requested while this potentially slow
                        # semantic call was running. Ignore its old box and let
                        # the next loop reacquire from the newest camera frame.
                        tracker.reset()
                        current_box = None
                        misses = 0
                        next_attempt_time = 0.0
                        self._update(
                            session,
                            state="REACQUIRING",
                            tracker=None,
                            model_calls=model_calls,
                            message="Ignored stale acquisition result after reset",
                        )
                        continue
                    if observation.success and observation.bbox_xyxy is not None:
                        try:
                            tracker.initialize(
                                working,
                                self._scale_box(observation.bbox_xyxy, scale),
                            )
                            current_box = observation.bbox_xyxy
                            misses = 0
                            failure_streak = 0
                            state = "TRACKING"
                            self._update(
                                session,
                                state="TRACKING",
                                backend=observation.backend or selected_backend or "auto",
                                tracker=tracker.active_type,
                                message="Target acquired",
                            )
                        except Exception as exc:
                            failure_streak += 1
                            next_attempt_time = time.monotonic() + min(30.0, 3.0 * (2 ** min(failure_streak, 3)))
                            self._update(
                                session,
                                state="ACQUIRING",
                                message=f"Tracker initialization failed: {exc}",
                            )
                    else:
                        failure_streak += 1
                        next_attempt_time = time.monotonic() + min(30.0, 3.0 * (2 ** min(failure_streak, 3)))
                        self._update(
                            session,
                            state="ACQUIRING",
                            message=observation.message or "No valid target box returned",
                        )
                elif current_box is not None:
                    ok, tracked_small = tracker.update(working)
                    if ok and tracked_small is not None:
                        current_box = self._unscale_box(
                            tracked_small,
                            scale,
                            frame.shape[1],
                            frame.shape[0],
                        )
                        misses = 0
                        state = "TRACKING"
                    else:
                        misses += 1
                        state = "UNCERTAIN"
                        if misses >= 8:
                            tracker.reset()
                            current_box = None
                            misses = 0
                            next_attempt_time = time.monotonic()
                            state = "REACQUIRING"

                annotated = frame.copy()
                if current_box is not None:
                    if session.use_segmentation:
                        try:
                            segmentation = segment_from_box(frame[:, :, ::-1], current_box)
                            draw_segmentation_overlay(annotated, segmentation.mask)
                        except Exception:
                            pass
                    self._draw_box(
                        annotated,
                        current_box,
                        f"{plan.target_object or session.instruction} | {state}",
                    )
                else:
                    cv2.putText(
                        annotated,
                        f"{state}: {plan.target_object or session.instruction}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                preview = self._preview_frame(annotated, 960)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    preview,
                    [cv2.IMWRITE_JPEG_QUALITY, 82],
                )
                if ok:
                    with session.lock:
                        session.latest_jpeg = encoded.tobytes()
                        session.frame_version += 1

                processed_since_rate += 1
                elapsed = time.perf_counter() - rate_started
                if elapsed >= 1.0:
                    fps = processed_since_rate / elapsed
                    processed_since_rate = 0
                    rate_started = time.perf_counter()
                    self._update(
                        session,
                        status="running",
                        state=state,
                        tracker=tracker.active_type,
                        model_calls=model_calls,
                        tracking_fps=round(fps, 2),
                    )
        except Exception as exc:
            self._update(
                session,
                status="failed",
                state="FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            session.stop_event.set()
            capture.release()
            capture_thread.join(timeout=1.0)
            service.close()
            if self.get(session.session_id) and self.get(session.session_id).get("status") != "failed":
                self._update(
                    session,
                    status="stopped",
                    state="STOPPED",
                    message="Live camera session stopped",
                    model_calls=model_calls,
                )

    @staticmethod
    def _select_backend(explicit: str | None, plan: VideoTaskPlan) -> str | None:
        value = normalize_text(explicit)
        if value in {"", "auto", "none"}:
            return plan.recommended_backend
        return value

    @staticmethod
    def _tracking_frame(frame, maximum_width: int):
        cv2 = LiveCameraSessionManager._cv2()
        height, width = frame.shape[:2]
        if width <= maximum_width:
            return frame, 1.0
        scale = maximum_width / float(width)
        return cv2.resize(
            frame,
            (maximum_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        ), scale

    @staticmethod
    def _preview_frame(frame, maximum_width: int):
        cv2 = LiveCameraSessionManager._cv2()
        height, width = frame.shape[:2]
        if width <= maximum_width:
            return frame
        scale = maximum_width / float(width)
        return cv2.resize(
            frame,
            (maximum_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _scale_box(box: Box, scale: float) -> Box:
        return tuple(float(value) * scale for value in box)  # type: ignore[return-value]

    @staticmethod
    def _unscale_box(box: Box, scale: float, width: int, height: int) -> Box:
        factor = 1.0 / scale if scale > 0 else 1.0
        x1, y1, x2, y2 = (float(value) * factor for value in box)
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(x1 + 1.0, min(float(width), x2))
        y2 = max(y1 + 1.0, min(float(height), y2))
        return x1, y1, x2, y2

    @staticmethod
    def _draw_box(frame, box: Box, label: str) -> None:
        cv2 = LiveCameraSessionManager._cv2()
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label[:100],
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _update(session: _LiveSession, **values: Any) -> None:
        with session.lock:
            session.status.update(values)
