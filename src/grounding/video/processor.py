"""Efficient recorded-video grounding with local target tracking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

from ..llm_task_parser import LLMVideoTaskParser, VideoTaskPlan
from ..task_parser import normalize_text
from ..segmentation import draw_segmentation_overlay, segment_from_box
from .service_adapter import GroundingObservation, HTTPGroundingServiceAdapter
from .tracker import OpenCVBoxTracker


Box = tuple[float, float, float, float]
ProgressCallback = Callable[[dict[str, Any], Any], None]


def _smooth_box(previous: Box | None, current: Box, alpha: float) -> Box:
    if previous is None:
        return current
    weight = max(0.0, min(1.0, float(alpha)))
    return tuple(
        (1.0 - weight) * old + weight * new
        for old, new in zip(previous, current)
    )  # type: ignore[return-value]


class VideoGroundingProcessor:
    """Acquire a semantic target once, then track it locally.

    Expensive grounding is used only for acquisition or sustained tracking loss.
    Tracking runs at a reduced working resolution while annotations are written
    on the original-resolution frame.
    """

    def __init__(
        self,
        service: HTTPGroundingServiceAdapter,
        *,
        output_root: str | Path = "runtime/video_sessions",
        tracker_type: str = "AUTO",
        lost_frames: int = 8,
        poor_quality_frames_to_reacquire: int = 3,
        acquisition_interval_frames: int = 30,
        maximum_acquisition_interval_frames: int = 120,
        smoothing_alpha: float = 0.55,
        tracker_confidence_decay: float = 0.999,
        minimum_tracker_quality: float = 0.45,
        jpeg_quality: int = 92,
        tracking_max_width: int = 1280,
        progress_interval_frames: int = 15,
        log_flush_interval_frames: int = 60,
        task_parser: LLMVideoTaskParser | None = None,
    ) -> None:
        self.service = service
        self.output_root = Path(output_root)
        self.tracker_type = tracker_type
        self.lost_frames = max(1, int(lost_frames))
        self.poor_quality_frames_to_reacquire = max(
            1,
            int(poor_quality_frames_to_reacquire),
        )
        self.acquisition_interval_frames = max(1, int(acquisition_interval_frames))
        self.maximum_acquisition_interval_frames = max(
            self.acquisition_interval_frames,
            int(maximum_acquisition_interval_frames),
        )
        self.smoothing_alpha = max(0.0, min(1.0, float(smoothing_alpha)))
        self.tracker_confidence_decay = max(
            0.0,
            min(1.0, float(tracker_confidence_decay)),
        )
        self.minimum_tracker_quality = max(
            0.0,
            min(1.0, float(minimum_tracker_quality)),
        )
        self.jpeg_quality = max(40, min(100, int(jpeg_quality)))
        self.tracking_max_width = max(320, int(tracking_max_width))
        self.progress_interval_frames = max(1, int(progress_interval_frames))
        self.log_flush_interval_frames = max(1, int(log_flush_interval_frames))
        self.task_parser = task_parser or LLMVideoTaskParser()

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("video support requires OpenCV") from exc
        return cv2

    def process(
        self,
        *,
        source: str | int,
        instruction: str,
        preferred_backend: str | None = None,
        maximum_latency_ms: int = 200_000,
        output_video: str | Path | None = None,
        save_video: bool = True,
        display: bool = False,
        maximum_frames: int | None = None,
        session_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
        use_llm_parser: bool = True,
        use_segmentation: bool = False,
        performance_mode: str = "balanced",
    ) -> dict[str, Any]:
        cv2 = self._cv2()
        plan = self.task_parser.parse(instruction, use_llm=use_llm_parser)
        selected_backend = self._select_backend(
            preferred_backend,
            plan,
            performance_mode,
        )

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"could not open video source: {source}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0.0 or source_fps > 240.0:
            source_fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            ok, probe = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError("video source opened but returned no frames")
            height, width = probe.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        session_id = session_name or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid4().hex[:8]
        )
        session_dir = (self.output_root / session_id).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = session_dir / "predictions.jsonl"
        summary_path = session_dir / "summary.json"

        final_annotated_path = (
            Path(output_video).expanduser().resolve()
            if output_video is not None
            else session_dir / "annotated_video.mp4"
        )
        final_annotated_path.parent.mkdir(parents=True, exist_ok=True)
        raw_annotated_path = session_dir / "annotated_raw.mp4"

        writer = None
        if save_video:
            writer = cv2.VideoWriter(
                str(raw_annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (width, height),
            )
            if not writer.isOpened():
                capture.release()
                raise RuntimeError(f"could not create output video: {raw_annotated_path}")

        tracker = OpenCVBoxTracker(self.tracker_type)
        current_box: Box | None = None
        current_confidence = 0.0
        current_backend: str | None = None
        state = "SEARCHING"
        misses = 0
        poor_quality_frames = 0
        model_calls = 0
        successful_acquisitions = 0
        reacquisitions = 0
        rejected_anchor_results = 0
        acquisition_failures = 0
        frames_processed = 0
        acquired_once = False
        next_model_frame = 0
        grounding_failure_streak = 0
        last_grounding_status: str | None = None
        last_grounding_error: str | None = None
        started = time.perf_counter()
        tracker_time_seconds = 0.0
        stopped_by_user = False

        try:
            with predictions_path.open("w", encoding="utf-8") as log_handle:
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if maximum_frames is not None and frame_index >= maximum_frames:
                        break

                    frame_started = time.perf_counter()
                    source_kind = "none"
                    model_latency_ms = 0.0
                    message = ""
                    tracker_quality = tracker.last_quality
                    model_called_this_frame = False
                    tracking_frame, tracking_scale = self._tracking_frame(frame)

                    if current_box is None:
                        if frame_index >= next_model_frame:
                            model_called_this_frame = True
                            observation, model_latency_ms = self._ground(
                                frame,
                                plan=plan,
                                preferred_backend=selected_backend,
                                maximum_latency_ms=maximum_latency_ms,
                                performance_mode=performance_mode,
                            )
                            model_calls += 1
                            last_grounding_status = observation.status
                            last_grounding_error = observation.message or None
                            message = observation.message

                            if self._is_anchor_only(observation, plan):
                                rejected_anchor_results += 1
                                observation.success = False
                                observation.message = (
                                    "anchor-only result rejected; target acquisition required"
                                )
                                message = observation.message
                                last_grounding_error = message

                            if observation.success and observation.bbox_xyxy is not None:
                                try:
                                    tracker.initialize(
                                        tracking_frame,
                                        self._scale_box(observation.bbox_xyxy, tracking_scale),
                                    )
                                except Exception as exc:
                                    acquisition_failures += 1
                                    grounding_failure_streak += 1
                                    last_grounding_error = (
                                        f"tracker initialization failed: {exc}"
                                    )
                                    message = last_grounding_error
                                    state = "SEARCHING" if not acquired_once else "LOST"
                                    next_model_frame = frame_index + self._retry_delay(
                                        grounding_failure_streak
                                    )
                                else:
                                    current_box = observation.bbox_xyxy
                                    current_confidence = observation.confidence
                                    current_backend = observation.backend
                                    state = "ACQUIRED" if not acquired_once else "REACQUIRED"
                                    if acquired_once:
                                        reacquisitions += 1
                                    acquired_once = True
                                    successful_acquisitions += 1
                                    misses = 0
                                    poor_quality_frames = 0
                                    grounding_failure_streak = 0
                                    source_kind = "grounding"
                            else:
                                acquisition_failures += 1
                                grounding_failure_streak += 1
                                state = "SEARCHING" if not acquired_once else "LOST"
                                next_model_frame = frame_index + self._retry_delay(
                                    grounding_failure_streak
                                )
                    else:
                        tracker_started = time.perf_counter()
                        track_ok, tracked_box_small = tracker.update(tracking_frame)
                        tracker_time_seconds += time.perf_counter() - tracker_started
                        tracker_quality = tracker.last_quality
                        if track_ok and tracked_box_small is not None:
                            tracked_box = self._unscale_box(
                                tracked_box_small,
                                tracking_scale,
                                width,
                                height,
                            )
                            current_confidence *= self.tracker_confidence_decay
                            source_kind = tracker.last_source
                            misses = 0
                            if tracker_quality < self.minimum_tracker_quality:
                                poor_quality_frames += 1
                                state = "UNCERTAIN"
                            else:
                                current_box = _smooth_box(
                                    current_box,
                                    tracked_box,
                                    self.smoothing_alpha,
                                )
                                poor_quality_frames = 0
                                state = "TRACKING"
                        else:
                            misses += 1
                            state = "UNCERTAIN"

                        if (
                            misses >= self.lost_frames
                            or poor_quality_frames
                            >= self.poor_quality_frames_to_reacquire
                        ):
                            tracker.reset()
                            current_box = None
                            current_confidence = 0.0
                            state = "LOST"
                            source_kind = "none"
                            poor_quality_frames = 0
                            grounding_failure_streak = 0
                            next_model_frame = frame_index + 1

                    annotated = frame.copy()
                    segmentation_payload = None
                    if current_box is not None:
                        if use_segmentation:
                            try:
                                segmentation = segment_from_box(frame[:, :, ::-1], current_box)
                                draw_segmentation_overlay(annotated, segmentation.mask)
                                segmentation_payload = segmentation.payload
                            except Exception:
                                segmentation_payload = None
                        self._draw_box(
                            annotated,
                            current_box,
                            label=(
                                f"{plan.target_object or instruction} | {state} | "
                                f"{current_backend or 'tracker'}"
                            ),
                        )
                    else:
                        cv2.putText(
                            annotated,
                            f"{state}: searching for {plan.target_object or instruction}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if writer is not None:
                        writer.write(annotated)

                    timestamp_seconds = frame_index / source_fps
                    frame_latency_ms = (time.perf_counter() - frame_started) * 1000.0
                    event = {
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp_seconds, 6),
                        "state": state,
                        "source": source_kind,
                        "bbox_xyxy": list(current_box) if current_box is not None else None,
                        "confidence": round(current_confidence, 6),
                        "backend": current_backend,
                        "requested_backend": selected_backend,
                        "tracker": tracker.active_type,
                        "tracker_quality": round(float(tracker_quality), 6),
                        "model_latency_ms": round(model_latency_ms, 3),
                        "frame_processing_latency_ms": round(frame_latency_ms, 3),
                        "message": message,
                        "segmentation": segmentation_payload,
                    }
                    log_handle.write(json.dumps(event) + "\n")
                    frames_processed += 1
                    if frames_processed % self.log_flush_interval_frames == 0:
                        log_handle.flush()

                    should_report = (
                        model_called_this_frame
                        or frame_index % self.progress_interval_frames == 0
                    )
                    if should_report and progress_callback is not None:
                        progress_callback(
                            {
                                "session_id": session_id,
                                "status": "processing",
                                "phase": "tracking" if current_box is not None else "acquiring",
                                "frame_index": frame_index,
                                "total_frames": total_source_frames,
                                "progress": (
                                    (frame_index + 1) / total_source_frames
                                    if total_source_frames > 0
                                    else None
                                ),
                                "state": state,
                                "backend": current_backend or selected_backend,
                                "tracker": tracker.active_type,
                                "tracker_quality": tracker_quality,
                                "model_calls": model_calls,
                                "successful_acquisitions": successful_acquisitions,
                                "last_grounding_status": last_grounding_status,
                                "last_grounding_error": last_grounding_error,
                                "plan": plan.model_dump(),
                            },
                            None,
                        )

                    if display:
                        cv2.imshow("MIE1077 Video Grounding", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            stopped_by_user = True
                            break

                    frame_index += 1
                log_handle.flush()
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        encoding = {
            "video_ready": False,
            "browser_video_compatible": False,
            "video_codec": None,
            "video_encoding_error": None,
        }
        if save_video:
            if progress_callback is not None:
                progress_callback(
                    {
                        "session_id": session_id,
                        "status": "processing",
                        "phase": "encoding",
                        "state": "ENCODING",
                        "frame_index": frames_processed,
                        "total_frames": total_source_frames,
                        "progress": 1.0,
                        "model_calls": model_calls,
                        "successful_acquisitions": successful_acquisitions,
                        "plan": plan.model_dump(),
                    },
                    None,
                )
            encoding = self._make_browser_video(
                raw_annotated_path,
                final_annotated_path,
            )

        elapsed = time.perf_counter() - started
        summary: dict[str, Any] = {
            "session_id": session_id,
            "status": "completed",
            "instruction": instruction,
            "plan": plan.model_dump(),
            "requested_backend": selected_backend,
            "performance_mode": performance_mode,
            "source": str(source),
            "source_fps": source_fps,
            "total_source_frames": total_source_frames,
            "frame_size": [width, height],
            "tracking_frame_max_width": self.tracking_max_width,
            "frames_processed": frames_processed,
            "elapsed_seconds": round(elapsed, 3),
            "average_processing_fps": (
                round(frames_processed / elapsed, 3) if elapsed > 0 else 0.0
            ),
            "tracker_update_fps": (
                round(frames_processed / tracker_time_seconds, 3)
                if tracker_time_seconds > 0
                else 0.0
            ),
            "model_calls": model_calls,
            "successful_acquisitions": successful_acquisitions,
            "reacquisitions": reacquisitions,
            "acquisition_failures": acquisition_failures,
            "anchor_rejections": rejected_anchor_results,
            "last_grounding_status": last_grounding_status,
            "last_grounding_error": last_grounding_error,
            "final_state": state,
            "tracker_requested": tracker.requested_type,
            "tracker_used": tracker.active_type,
            "minimum_tracker_quality": self.minimum_tracker_quality,
            "poor_quality_frames_to_reacquire": (
                self.poor_quality_frames_to_reacquire
            ),
            "smoothing_alpha": self.smoothing_alpha,
            "annotated_video_path": (
                str(final_annotated_path) if save_video and final_annotated_path.exists() else None
            ),
            "predictions_path": str(predictions_path),
            "summary_path": str(summary_path),
            "termination_reason": "user" if stopped_by_user else "end_of_stream",
            "scope": "single_primary_target",
            "use_segmentation": bool(use_segmentation),
            **encoding,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if progress_callback is not None:
            progress_callback(summary, None)
        return summary

    def _retry_delay(self, failure_streak: int) -> int:
        multiplier = 2 ** min(max(0, failure_streak - 1), 3)
        return min(
            self.maximum_acquisition_interval_frames,
            self.acquisition_interval_frames * multiplier,
        )

    @staticmethod
    def _select_backend(
        explicit: str | None,
        plan: VideoTaskPlan,
        performance_mode: str = "balanced",
    ) -> str | None:
        value = normalize_text(explicit)
        if value in {"", "auto", "none"}:
            value = ""
        if value:
            return value
        if normalize_text(performance_mode) in {"quality", "accurate"}:
            return "gpt_guided_owlvit"
        return plan.recommended_backend

    def _ground(
        self,
        frame,
        *,
        plan: VideoTaskPlan,
        preferred_backend: str | None,
        maximum_latency_ms: int,
        performance_mode: str,
    ) -> tuple[GroundingObservation, float]:
        started = time.perf_counter()
        observation = self.service.ground_frame(
            frame,
            instruction=plan.instruction,
            target_object=plan.target_object,
            location_hint=plan.location_hint,
            action="find",
            performance_mode=performance_mode,
            preferred_backend=preferred_backend,
            maximum_latency_ms=maximum_latency_ms,
            jpeg_quality=self.jpeg_quality,
        )
        return observation, (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _is_anchor_only(observation: GroundingObservation, plan: VideoTaskPlan) -> bool:
        if not observation.label or not plan.anchor_objects:
            return False
        label = normalize_text(observation.label)
        target = normalize_text(plan.target_object)
        if target and target in label:
            return False
        return any(
            normalize_text(anchor) == label or normalize_text(anchor) in label
            for anchor in plan.anchor_objects
        )

    def _tracking_frame(self, frame):
        cv2 = self._cv2()
        height, width = frame.shape[:2]
        if width <= self.tracking_max_width:
            return frame, 1.0
        scale = self.tracking_max_width / float(width)
        resized = cv2.resize(
            frame,
            (self.tracking_max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _scale_box(box: Box, scale: float) -> Box:
        return tuple(float(value) * scale for value in box)  # type: ignore[return-value]

    @staticmethod
    def _unscale_box(
        box: Box,
        scale: float,
        frame_width: int,
        frame_height: int,
    ) -> Box:
        if scale <= 0:
            scale = 1.0
        values = tuple(float(value) / scale for value in box)
        x1, y1, x2, y2 = values
        return (
            max(0.0, min(float(frame_width - 1), x1)),
            max(0.0, min(float(frame_height - 1), y1)),
            max(x1 + 1.0, min(float(frame_width), x2)),
            max(y1 + 1.0, min(float(frame_height), y2)),
        )

    @staticmethod
    def _draw_box(frame, box: Box, *, label: str) -> None:
        cv2 = VideoGroundingProcessor._cv2()
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
    def _make_browser_video(raw_path: Path, final_path: Path) -> dict[str, Any]:
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            return {
                "video_ready": False,
                "browser_video_compatible": False,
                "video_codec": None,
                "video_encoding_error": "raw annotated video was not created",
            }

        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(raw_path),
                "-an",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
                str(final_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if (
                completed.returncode != 0
                or not final_path.exists()
                or final_path.stat().st_size < 1024
            ):
                raise RuntimeError(completed.stderr[-2000:] or "FFmpeg conversion failed")
            raw_path.unlink(missing_ok=True)
            return {
                "video_ready": True,
                "browser_video_compatible": True,
                "video_codec": "h264",
                "video_encoding_error": None,
            }
        except Exception as exc:
            try:
                shutil.copy2(raw_path, final_path)
            except Exception:
                pass
            return {
                "video_ready": final_path.exists() and final_path.stat().st_size > 0,
                "browser_video_compatible": False,
                "video_codec": "mp4v",
                "video_encoding_error": f"{type(exc).__name__}: {exc}",
            }
