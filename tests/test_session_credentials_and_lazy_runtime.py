from pathlib import Path

from grounding.backends.gpt_guided_owlvit_backend import GPTGuidedOWLViTBackend
from grounding.openai_context import resolve_openai_api_key, use_openai_api_key
from grounding.streamlit_ui.runtime import ServiceRuntime


class _LoadedHealth:
    loaded = True


class _LoadedOwl:
    def health(self):
        return _LoadedHealth()


def test_session_api_key_is_scoped_and_restored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    assert resolve_openai_api_key() == "server-key"

    with use_openai_api_key("session-key"):
        assert resolve_openai_api_key() == "session-key"

    assert resolve_openai_api_key() == "server-key"


def test_gpt_backend_health_requires_key_for_session_mode(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = GPTGuidedOWLViTBackend(
        owlvit_backend=_LoadedOwl(),
        openai_model="test-model",
        allow_session_api_key=True,
    )
    backend.startup()
    assert backend.health().loaded is False

    with use_openai_api_key("session-key"):
        assert backend.health().loaded is True


def test_streamlit_runtime_defers_model_startup_and_selects_by_instruction():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "grounding_service.json"
    )
    runtime = ServiceRuntime.from_config(config_path)

    assert all(
        not getattr(backend, "_started", False)
        for backend in runtime.service.backends.values()
    )
    assert runtime.select_backend(None, "find the person") == "yolo"
    assert runtime.select_backend(None, "find the soldering iron") == "owlvit"
    assert (
        runtime.select_backend(
            None,
            "find the green bottle beside the chair",
            api_key="session-key",
        )
        == "gpt_guided_owlvit"
    )
