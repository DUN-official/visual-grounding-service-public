from types import SimpleNamespace

from grounding.video import service_adapter
from grounding.video.service_adapter import LocalGroundingServiceAdapter


class _Service:
    def __init__(self):
        self.request = None

    def ground(self, request):
        self.request = request
        payload = {
            "request_id": request.request_id,
            "status": "success",
            "bbox_xyxy": {"x_min": 1, "y_min": 2, "x_max": 20, "y_max": 30},
            "predictions": [
                {
                    "bbox_xyxy": {"x_min": 1, "y_min": 2, "x_max": 20, "y_max": 30},
                    "confidence": 0.9,
                    "label": "package",
                }
            ],
            "confidence": 0.9,
            "backend_used": "owlvit",
            "trace": [],
        }
        return SimpleNamespace(model_dump=lambda **kwargs: payload)


def test_local_adapter_builds_frame_request(monkeypatch):
    monkeypatch.setattr(service_adapter, "_encode_frame", lambda frame, quality: b"jpeg")
    service = _Service()
    adapter = LocalGroundingServiceAdapter(service)

    observation = adapter.ground_frame(
        object(),
        instruction="find the package",
        preferred_backend="owlvit",
    )

    assert observation.success is True
    assert observation.selected_label == "package"
    assert observation.bbox_xyxy == (1.0, 2.0, 20.0, 30.0)
    assert service.request.preferred_backend == "owlvit"
    assert service.request.metadata["input_mode"] == "video_frame"


def test_local_adapter_propagates_profile_without_reparsing(monkeypatch):
    monkeypatch.setattr(service_adapter, "_encode_frame", lambda frame, quality: b"jpeg")
    prepared = {}

    def prepare_backend(preferred, instruction, **kwargs):
        prepared.update(
            preferred=preferred,
            instruction=instruction,
            **kwargs,
        )

    service = _Service()
    adapter = LocalGroundingServiceAdapter(
        service,
        api_key="session-key",
        prepare_backend=prepare_backend,
    )

    adapter.ground_frame(
        object(),
        instruction="find the package beside the chair",
        preferred_backend="gpt_guided_owlvit",
        performance_mode="quality",
    )

    assert service.request.performance_mode == "quality"
    assert service.request.metadata["parser_mode"] == "local"
    assert prepared["performance_mode"] == "quality"
    assert prepared["api_key"] == "session-key"
