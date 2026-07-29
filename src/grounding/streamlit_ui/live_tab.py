"""Browser-camera grounding through WebRTC."""

from __future__ import annotations

from threading import RLock
import time

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from ..segmentation import draw_segmentation_overlay, segment_from_box
from ..schemas import PerformanceMode
from ..video.tracker import OpenCVBoxTracker
from .image_tab import BACKENDS, MODE_LATENCY_MS, PERFORMANCE_MODES
from .runtime import ServiceRuntime


def _smooth_box(previous, current, alpha=0.55):
    if previous is None:
        return current
    return tuple(
        (1.0 - alpha) * old + alpha * new
        for old, new in zip(previous, current)
    )


class LiveFrameController:
    def __init__(self, runtime: ServiceRuntime, *, api_key: str | None = None) -> None:
        self.adapter = runtime.video_adapter(api_key=api_key)
        self.api_key = api_key
        self.lock = RLock()
        self.tracker = OpenCVBoxTracker("AUTO")
        self.instruction = ""
        self.backend: str | None = None
        self.performance_mode = PerformanceMode.QUALITY
        self.use_segmentation = False
        self.current_box = None
        self.state = "IDLE"
        self.frame_index = 0
        self.next_grounding_frame = 0
        self.misses = 0
        self.poor_quality_frames = 0
        self.minimum_tracker_quality = 0.45
        self.poor_quality_limit = 3
        self.miss_limit = 4
        self.confidence = 0.0
        self.tracker_quality = 0.0
        self.tracker_source = "none"
        self.backend_used: str | None = None
        self.fps = 0.0
        self._last_frame_time: float | None = None

    def configure(
        self,
        *,
        instruction: str,
        backend: str | None,
        performance_mode: PerformanceMode,
        use_segmentation: bool,
    ) -> None:
        with self.lock:
            changed = (instruction, backend, performance_mode) != (
                self.instruction,
                self.backend,
                self.performance_mode,
            )
            self.instruction = instruction
            self.backend = backend
            self.performance_mode = performance_mode
            self.use_segmentation = use_segmentation
            if changed:
                self._reset_locked("REACQUIRING")

    def request_reset(self) -> None:
        with self.lock:
            self._reset_locked("REACQUIRING")

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "backend": self.backend_used or self.backend or "auto",
                "profile": self.performance_mode.value,
                "tracker": self.tracker.active_type or "acquiring",
                "tracking_fps": round(self.fps, 1),
                "confidence": round(self.confidence, 3),
                "tracker_quality": round(self.tracker_quality, 3),
                "tracker_source": self.tracker_source,
            }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated = self._process(image)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _process(self, frame):
        import cv2

        with self.lock:
            now = time.perf_counter()
            if self._last_frame_time is not None:
                interval = max(1e-6, now - self._last_frame_time)
                instantaneous = 1.0 / interval
                self.fps = instantaneous if self.fps == 0 else 0.85 * self.fps + 0.15 * instantaneous
            self._last_frame_time = now

            if not self.instruction:
                self.state = "WAITING_FOR_INSTRUCTION"
            elif self.current_box is None and self.frame_index >= self.next_grounding_frame:
                self.state = "ACQUIRING"
                observation = self.adapter.ground_frame(
                    frame,
                    instruction=self.instruction,
                    preferred_backend=self.backend,
                    performance_mode=self.performance_mode.value,
                    maximum_latency_ms=MODE_LATENCY_MS[self.performance_mode],
                    jpeg_quality=92 if self.performance_mode == PerformanceMode.QUALITY else 90,
                )
                if observation.success and observation.bbox_xyxy is not None:
                    self.tracker.initialize(frame, observation.bbox_xyxy)
                    self.current_box = observation.bbox_xyxy
                    self.confidence = observation.confidence
                    self.backend_used = observation.backend
                    self.misses = 0
                    self.poor_quality_frames = 0
                    self.tracker_quality = 1.0
                    self.tracker_source = "grounding"
                    self.state = "ACQUIRED"
                else:
                    self.state = "SEARCHING"
                    self.next_grounding_frame = self.frame_index + 15
            elif self.current_box is not None:
                ok, box = self.tracker.update(frame)
                self.tracker_quality = self.tracker.last_quality
                self.tracker_source = self.tracker.last_source
                if ok and box is not None:
                    self.misses = 0
                    if self.tracker_quality >= self.minimum_tracker_quality:
                        self.current_box = _smooth_box(self.current_box, box)
                        self.poor_quality_frames = 0
                        self.confidence *= 0.999
                        self.state = "TRACKING"
                    else:
                        self.poor_quality_frames += 1
                        self.state = "UNCERTAIN"
                else:
                    self.misses += 1
                    self.state = "UNCERTAIN"
                if (
                    self.misses >= self.miss_limit
                    or self.poor_quality_frames >= self.poor_quality_limit
                ):
                    self._reset_locked("LOST")

            annotated = frame.copy()
            if self.current_box is not None:
                if self.use_segmentation:
                    try:
                        segmentation = segment_from_box(
                            frame[:, :, ::-1],
                            self.current_box,
                        )
                        draw_segmentation_overlay(annotated, segmentation.mask)
                    except Exception:
                        pass
                x1, y1, x2, y2 = [int(round(value)) for value in self.current_box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 176, 255), 2)
                cv2.putText(
                    annotated,
                    f"{self.state} | {self.backend_used or self.backend or 'auto'}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 176, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    annotated,
                    self.state,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 176, 255),
                    2,
                    cv2.LINE_AA,
                )
            self.frame_index += 1
            return annotated

    def _reset_locked(self, state: str) -> None:
        self.tracker.reset()
        self.current_box = None
        self.confidence = 0.0
        self.backend_used = None
        self.misses = 0
        self.poor_quality_frames = 0
        self.tracker_quality = 0.0
        self.tracker_source = "none"
        self.next_grounding_frame = self.frame_index
        self.state = state


@st.fragment(run_every=1.0)
def _render_metrics(controller: LiveFrameController) -> None:
    metrics = controller.snapshot()
    columns = st.columns(6)
    columns[0].metric("Tracking state", metrics["state"])
    columns[1].metric("Profile", metrics["profile"].title())
    columns[2].metric("Backend", metrics["backend"])
    columns[3].metric("Tracker", metrics["tracker"])
    columns[4].metric("Tracking FPS", metrics["tracking_fps"])
    columns[5].metric("Tracker quality", metrics["tracker_quality"])
    st.caption(f"Tracker source: {metrics['tracker_source']}")


def render_live_tab(runtime: ServiceRuntime, *, api_key: str | None = None) -> None:
    instruction = st.text_input(
        "Instruction",
        placeholder="Track the package beside the table",
        key="live-instruction",
    )
    mode_label = st.selectbox(
        "Profile",
        list(PERFORMANCE_MODES),
        help="The profile applies to acquisition and re-acquisition frames.",
        key="live-performance-mode",
    )
    backend_label = st.selectbox("Backend", list(BACKENDS), key="live-backend")
    segmentation = st.checkbox(
        "Overlay segmentation mask on the live camera stream",
        key="live-segmentation",
    )
    st.caption(
        "Segmentation is experimental and significantly reduces live performance."
    )

    selected_backend = BACKENDS[backend_label]
    performance_mode = PERFORMANCE_MODES[mode_label]
    if performance_mode == PerformanceMode.QUALITY and selected_backend not in {
        None,
        "gpt_guided_owlvit",
    }:
        st.warning("Quality profile requires Auto or GPT-guided OWL-ViT as the backend.")
        return
    if (
        performance_mode == PerformanceMode.QUALITY
        or selected_backend == "gpt_guided_owlvit"
    ) and not api_key:
        st.warning("Enter an OpenAI API key in the sidebar before starting GPT-guided live grounding.")
        return

    controller = st.session_state.get("live-controller")
    if (
        not isinstance(controller, LiveFrameController)
        or controller.api_key != api_key
    ):
        controller = LiveFrameController(runtime, api_key=api_key)
        st.session_state["live-controller"] = controller
    controller.configure(
        instruction=instruction.strip(),
        backend=selected_backend,
        performance_mode=performance_mode,
        use_segmentation=segmentation,
    )

    if st.button("Reset and re-acquire", width="stretch"):
        controller.request_reset()

    st.caption(
        "Keep the camera stable during initial grounding. Use START and STOP in "
        "the camera panel to control capture."
    )
    webrtc_streamer(
        key="mie1077-live-grounding",
        video_frame_callback=controller.recv,
        media_stream_constraints={"video": True, "audio": False},
        media_toggle_controls=False,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )
    _render_metrics(controller)
