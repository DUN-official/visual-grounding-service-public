"""Image-grounding view."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image
import streamlit as st

from ..schemas import GroundingRequest, ImagePayload, PerformanceMode, QuantityIntent
from .rendering import annotate_image, image_bytes, result_json
from .runtime import ServiceRuntime


BACKENDS = {
    "Auto": None,
    "YOLO": "yolo",
    "OWL-ViT": "owlvit",
    "GPT-guided OWL-ViT": "gpt_guided_owlvit",
}

PERFORMANCE_MODES = {
    "Quality": PerformanceMode.QUALITY,
    "Balanced": PerformanceMode.BALANCED,
    "Fast": PerformanceMode.FAST,
}

MODE_LATENCY_MS = {
    PerformanceMode.QUALITY: 300_000,
    PerformanceMode.BALANCED: 200_000,
    PerformanceMode.FAST: 120_000,
}


def render_image_tab(runtime: ServiceRuntime, *, api_key: str | None = None) -> None:
    with st.form("image-grounding-form"):
        uploaded = st.file_uploader(
            "Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="image-upload",
        )
        instruction = st.text_input(
            "Instruction",
            placeholder="Find the package beside the table",
            key="image-instruction",
        )
        mode_label = st.selectbox(
            "Profile",
            list(PERFORMANCE_MODES),
            help=(
                "Quality uses multi-stage guided localization. Balanced limits "
                "remote review, while Fast prioritizes local inference."
            ),
            key="image-performance-mode",
        )
        backend_label = st.selectbox("Backend", list(BACKENDS), key="image-backend")
        maximum_results = st.slider(
            "Maximum results",
            min_value=1,
            max_value=10,
            value=1,
            key="image-maximum-results",
        )
        segmentation = st.checkbox(
            "Return segmentation mask overlay",
            key="image-segmentation",
        )
        submitted = st.form_submit_button("Run image grounding", width="stretch")

    if submitted:
        if uploaded is None:
            st.error("Choose an image before running grounding.")
            return
        if not instruction.strip():
            st.error("Enter an instruction before running grounding.")
            return

        performance_mode = PERFORMANCE_MODES[mode_label]
        if performance_mode == PerformanceMode.QUALITY and BACKENDS[backend_label] not in {
            None,
            "gpt_guided_owlvit",
        }:
            st.error("Quality profile requires Auto or GPT-guided OWL-ViT as the backend.")
            return
        if (
            performance_mode == PerformanceMode.QUALITY
            or BACKENDS[backend_label] == "gpt_guided_owlvit"
        ) and not api_key:
            st.error("Enter an OpenAI API key in the sidebar to use GPT-guided OWL-ViT.")
            return

        payload = uploaded.getvalue()
        media_type = uploaded.type or "image/jpeg"
        request = GroundingRequest(
            image=ImagePayload(
                base64_data=base64.b64encode(payload).decode("ascii"),
                media_type=media_type,
            ),
            instruction=instruction.strip(),
            performance_mode=performance_mode,
            maximum_results=maximum_results,
            quantity=(
                QuantityIntent.MULTIPLE
                if maximum_results > 1
                else QuantityIntent.UNKNOWN
            ),
            maximum_latency_ms=MODE_LATENCY_MS[performance_mode],
            preferred_backend=BACKENDS[backend_label],
            metadata={
                "uploaded_filename": uploaded.name,
                "input_mode": "streamlit_upload",
                "parser_mode": "llm",
                "return_segmentation": segmentation,
            },
        )
        with st.spinner("Preparing the selected model and running grounding..."):
            result = runtime.ground(request, api_key=api_key)
        source_image = Image.open(BytesIO(payload)).convert("RGB")
        rendered = annotate_image(source_image, result)
        st.session_state["image-result"] = (rendered, result)

    stored = st.session_state.get("image-result")
    if not stored:
        return
    rendered, result = stored
    st.image(rendered, caption="Annotated result", width="stretch")
    if result.status.value != "success":
        st.warning(result.error or "No grounded target was returned.")
    st.subheader("Structured result")
    st.json(result.model_dump(mode="json"))
    with st.expander("Pipeline diagnostics"):
        parser_event = next(
            (
                event.model_dump(mode="json")
                for event in result.trace
                if event.stage == "prompt_parser"
            ),
            None,
        )
        st.json(
            {
                "backend_used": result.backend_used,
                "parser": parser_event,
                "profile": result.metadata.get("performance_profile"),
                "pipeline_path": result.metadata.get("pipeline_path"),
                "stage_latencies_ms": result.metadata.get("stage_latencies_ms"),
                "gpt_request_count": result.metadata.get("gpt_request_count"),
            }
        )
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download annotated image",
            data=image_bytes(rendered),
            file_name="annotated_grounding.png",
            mime="image/png",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download result JSON",
            data=result_json(result),
            file_name="grounding_result.json",
            mime="application/json",
            width="stretch",
        )
