"""Recorded-video grounding view."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from ..openai_context import use_openai_api_key
from ..video.processor import VideoGroundingProcessor
from .image_tab import BACKENDS
from .runtime import ServiceRuntime


ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def render_video_tab(runtime: ServiceRuntime, *, api_key: str | None = None) -> None:
    with st.form("recorded-video-form"):
        uploaded = st.file_uploader(
            "Video",
            type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
            key="video-upload",
        )
        instruction = st.text_input(
            "Instruction",
            placeholder="Track the package beside the table",
            key="video-instruction",
        )
        backend_label = st.selectbox("Backend", list(BACKENDS), key="video-backend")
        segmentation = st.checkbox(
            "Overlay segmentation mask in the annotated output video",
            key="video-segmentation",
        )
        st.caption(
            "Experimental: segmentation significantly slows video processing and is "
            "not recommended for demonstrations."
        )
        submitted = st.form_submit_button("Start processing", width="stretch")

    if submitted:
        if uploaded is None:
            st.error("Choose a video before processing.")
            return
        if not instruction.strip():
            st.error("Enter an instruction before processing.")
            return
        if BACKENDS[backend_label] == "gpt_guided_owlvit" and not api_key:
            st.error("Enter an OpenAI API key in the sidebar to use GPT-guided OWL-ViT.")
            return
        _process_video(
            runtime,
            uploaded=uploaded,
            instruction=instruction.strip(),
            backend=BACKENDS[backend_label],
            segmentation=segmentation,
            api_key=api_key,
        )

    result = st.session_state.get("video-result")
    if result:
        _render_result(result)


def _process_video(
    runtime: ServiceRuntime,
    *,
    uploaded,
    instruction: str,
    backend: str | None,
    segmentation: bool,
    api_key: str | None,
) -> None:
    upload_root = runtime.project_root / "runtime" / "streamlit_uploads"
    output_root = runtime.project_root / "runtime" / "streamlit_sessions"
    upload_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".mp4"
    session_id = uuid4().hex[:16]
    upload_path = upload_root / f"{session_id}{suffix}"
    upload_path.write_bytes(uploaded.getvalue())

    progress = st.progress(0.0, text="Preparing video")
    status = st.empty()

    def report(data, _frame) -> None:
        value = data.get("progress")
        if value is None:
            value = 0.0
        value = max(0.0, min(1.0, float(value)))
        phase = str(data.get("phase") or data.get("state") or "processing")
        progress.progress(value, text=phase.replace("_", " ").title())
        status.caption(
            f"Frame {data.get('frame_index', 0)} | "
            f"Backend: {data.get('backend') or backend or 'auto'} | "
            f"Tracker: {data.get('tracker') or 'acquiring'}"
        )

    adapter = runtime.video_adapter(api_key=api_key)
    try:
        processor = VideoGroundingProcessor(
            adapter,
            output_root=output_root,
            tracker_type="AUTO",
            maximum_acquisition_interval_frames=120,
            tracking_max_width=1280,
            progress_interval_frames=15,
        )
        with use_openai_api_key(api_key):
            summary = processor.process(
                source=str(upload_path),
                instruction=instruction,
                preferred_backend=backend,
                maximum_latency_ms=200_000,
                save_video=True,
                display=False,
                session_name=session_id,
                progress_callback=report,
                use_llm_parser=True,
                use_segmentation=segmentation,
            )
        progress.progress(1.0, text="Completed")
        st.session_state["video-result"] = summary
    except Exception as exc:
        st.session_state.pop("video-result", None)
        st.error(f"Video processing failed: {type(exc).__name__}: {exc}")
    finally:
        adapter.close()
        upload_path.unlink(missing_ok=True)


def _render_result(summary: dict) -> None:
    video_path_value = summary.get("annotated_video_path")
    summary_path_value = summary.get("summary_path")
    predictions_path_value = summary.get("predictions_path")
    video_path = Path(video_path_value) if video_path_value else None
    summary_path = Path(summary_path_value) if summary_path_value else None
    predictions_path = Path(predictions_path_value) if predictions_path_value else None

    st.subheader("Annotated video")
    if video_path and video_path.exists():
        st.video(str(video_path))
    else:
        st.warning(summary.get("video_encoding_error") or "Annotated video is unavailable.")

    st.subheader("Processing summary")
    st.json(summary)
    columns = st.columns(3)
    with columns[0]:
        if video_path and video_path.exists():
            st.download_button(
                "Download video",
                data=video_path.read_bytes(),
                file_name="annotated_video.mp4",
                mime="video/mp4",
                width="stretch",
            )
    with columns[1]:
        summary_text = (
            summary_path.read_text(encoding="utf-8")
            if summary_path and summary_path.exists()
            else json.dumps(summary, indent=2)
        )
        st.download_button(
            "Download summary",
            data=summary_text,
            file_name="summary.json",
            mime="application/json",
            width="stretch",
        )
    with columns[2]:
        if predictions_path and predictions_path.exists():
            st.download_button(
                "Download predictions",
                data=predictions_path.read_bytes(),
                file_name="predictions.jsonl",
                mime="application/x-ndjson",
                width="stretch",
            )
