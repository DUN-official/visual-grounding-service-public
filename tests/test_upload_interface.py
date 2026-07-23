from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from grounding.evaluation.logger import GroundingJSONLLogger
from grounding.router import GroundingRouter, RoutingPolicy
from grounding.services import api_service
from grounding.services.local_service import LocalGroundingService

from conftest import FakeGroundingBackend


def test_upload_image_and_prompt_runs_pipeline(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "bag_scene.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)

    service = LocalGroundingService(
        backends={"owlvit": FakeGroundingBackend("owlvit")},
        router=GroundingRouter(RoutingPolicy(remote_fallback_enabled=False)),
        logger=GroundingJSONLLogger(tmp_path / "requests.jsonl"),
    )

    class FakeConfig:
        class Service:
            max_image_bytes = 1024 * 1024
        service = Service()

    monkeypatch.setattr(api_service, "load_config", lambda *args, **kwargs: FakeConfig())
    monkeypatch.setattr(
        LocalGroundingService,
        "from_config",
        classmethod(lambda cls, config: service),
    )

    app = api_service.create_app(tmp_path / "unused.json")
    with TestClient(app) as client:
        with image_path.open("rb") as handle:
            response = client.post(
                "/v1/ground/upload",
                data={"instruction": "find the bag on the table"},
                files={"image": (image_path.name, handle, "image/jpeg")},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["bbox_xyxy"] == {
        "x_min": 10.0,
        "y_min": 20.0,
        "x_max": 70.0,
        "y_max": 80.0,
    }
    assert payload["trace"][0]["data"]["target_object"] == "bag"
