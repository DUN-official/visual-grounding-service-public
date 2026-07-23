from pathlib import Path

from fastapi.testclient import TestClient

from grounding.evaluation.logger import GroundingJSONLLogger
from grounding.router import GroundingRouter, RoutingPolicy
from grounding.services import api_service
from grounding.services.local_service import LocalGroundingService

from conftest import FakeGroundingBackend


def test_task_parse_endpoint_uses_local_fallback_when_gpt_unavailable(
    tmp_path: Path, monkeypatch
):
    service = LocalGroundingService(
        backends={"owlvit": FakeGroundingBackend("owlvit")},
        router=GroundingRouter(RoutingPolicy(remote_fallback_enabled=False)),
        logger=GroundingJSONLLogger(tmp_path / "requests.jsonl"),
    )

    class FakeConfig:
        class Service:
            max_image_bytes = 1024 * 1024
            project_root = str(tmp_path)
        service = Service()

    monkeypatch.setattr(api_service, "load_config", lambda *args, **kwargs: FakeConfig())
    monkeypatch.setattr(
        LocalGroundingService,
        "from_config",
        classmethod(lambda cls, config: service),
    )

    app = api_service.create_app(tmp_path / "unused.json")
    with TestClient(app) as client:
        response = client.post(
            "/v1/task/parse",
            json={
                "instruction": "track toy car beside green water bottle",
                "performance_mode": "balanced",
                "parser_mode": "llm",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_object"] == "toy car"
    assert payload["anchor_objects"] == ["green water bottle"]
    assert payload["parser_source"] == "local_fallback"
    assert payload["requested_backend"] == "gpt_guided_owlvit"
