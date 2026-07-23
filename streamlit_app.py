"""Public Streamlit application for the robot-vision grounding project."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from grounding.streamlit_ui.image_tab import render_image_tab
from grounding.streamlit_ui.live_tab import render_live_tab
from grounding.streamlit_ui.runtime import ServiceRuntime
from grounding.streamlit_ui.video_tab import render_video_tab


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "grounding_service.json"
BACKEND_LABELS = {
    "yolo": "YOLO",
    "owlvit": "OWL-ViT",
    "gpt_guided_owlvit": "GPT-guided OWL-ViT",
}


st.set_page_config(
    page_title="Robot Vision Grounding",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_runtime(config_path: str) -> ServiceRuntime:
    return ServiceRuntime.from_config(config_path)


def _clear_session_key() -> None:
    st.session_state["openai-api-key"] = ""
    st.session_state.pop("live-controller", None)


def _server_api_key() -> str | None:
    environment_key = os.environ.get("OPENAI_API_KEY")
    if environment_key:
        return environment_key.strip()
    try:
        secret = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret = None
    return str(secret).strip() if secret else None


def _render_api_key_panel() -> tuple[str | None, str]:
    st.subheader("GPT-guided grounding")
    entered_key = st.text_input(
        "OpenAI API key",
        type="password",
        key="openai-api-key",
        placeholder="Required only for GPT-guided OWL-ViT",
        help=(
            "The key is held in this Streamlit session only and is never written "
            "to project files or logs."
        ),
    ).strip()
    st.button(
        "Clear session key",
        on_click=_clear_session_key,
        disabled=not bool(entered_key),
        width="stretch",
    )
    server_key = _server_api_key()
    if entered_key:
        st.success("GPT-guided OWL-ViT enabled with your session key.")
        return entered_key, "session"
    if server_key:
        st.success("GPT-guided OWL-ViT enabled by the deployment.")
        return server_key, "server"
    st.caption("Enter a key to enable GPT-guided OWL-ViT. Other backends remain available.")
    return None, "unavailable"


def _render_project_context() -> None:
    st.title("Robot Vision: Adaptive Visual Grounding")
    st.markdown(
        "This project developed as a natural offshoot of building robot-vision "
        "capabilities: converting a human instruction into a localized visual "
        "target that a robotic system can attend to, track, and eventually act on."
    )
    st.caption(
        "Compare local and GPT-guided grounding on images, recorded video, "
        "and a browser camera."
    )


def _render_limitations() -> None:
    with st.expander("Limitations and future work"):
        st.markdown(
            """
**Current limitations**

- Still-image grounding is the primary focus. Video bounding boxes can be less precise because detections are combined with tracking between inference cycles.
- Keep the camera stable during initial grounding. Initial inference can take several seconds on CPU or shared cloud compute.
- Clear camera quality, appropriate lighting, limited motion blur, and an unobstructed target improve results.
- Segmentation is experimental and is not recommended for recorded or live video demonstrations because it significantly reduces performance.
- This is a research prototype and is not intended for safety-critical robot control.

**Future work**

- Dedicated cloud inference backend
- ROS or ROS 2 deployment package
- Faster live inference through GPU execution, quantization, and asynchronous processing
- Improved video-box stability, target reacquisition, and real-time segmentation
"""
        )


def main() -> None:
    _render_project_context()

    config_path = Path(
        os.environ.get("GROUNDING_CONFIG", str(DEFAULT_CONFIG))
    ).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    try:
        runtime = load_runtime(str(config_path))
    except Exception as exc:
        st.error(f"The application could not start: {type(exc).__name__}: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Robot-vision controls")
        api_key, key_source = _render_api_key_panel()
        st.divider()
        st.subheader("Backend status")
        for name, health in runtime.health(api_key=api_key).items():
            status = str(health.get("status", "unknown"))
            detail = str(health.get("detail", ""))
            if detail == "not started":
                status = "loads on first use"
            st.write(f"**{BACKEND_LABELS.get(name, name)}** — {status}")
        st.caption(
            "Models download from their original providers when first selected. "
            "The first run may take several minutes."
        )
        if key_source == "session":
            st.caption("Session keys are cleared when the session ends.")

    image_tab, video_tab, live_tab = st.tabs(
        ["Image", "Recorded video", "Live camera"]
    )
    with image_tab:
        render_image_tab(runtime, api_key=api_key)
    with video_tab:
        render_video_tab(runtime, api_key=api_key)
    with live_tab:
        render_live_tab(runtime, api_key=api_key)

    _render_limitations()


if __name__ == "__main__":
    main()
