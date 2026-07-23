from grounding.evaluation.logger import GroundingJSONLLogger
from grounding.router import GroundingRouter, RoutingPolicy
from grounding.schemas import GroundingRequest, GroundingStatus, ImagePayload
from grounding.services.local_service import LocalGroundingService

from conftest import FakeGroundingBackend


def test_service_parses_prompt_routes_and_returns_result(tmp_path):
    service = LocalGroundingService(
        backends={"owlvit": FakeGroundingBackend("owlvit")},
        router=GroundingRouter(RoutingPolicy(remote_fallback_enabled=False)),
        logger=GroundingJSONLLogger(tmp_path / "requests.jsonl"),
    )
    service.startup()
    try:
        result = service.ground(
            GroundingRequest(
                image=ImagePayload(base64_data="aGVsbG8=", media_type="image/jpeg"),
                instruction="find the bag on the table",
            )
        )
    finally:
        service.shutdown()

    assert result.status == GroundingStatus.SUCCESS
    assert result.backend_used == "owlvit"
    assert result.trace[0].stage == "prompt_parser"
    assert result.trace[0].data["target_object"] == "bag"
    assert result.trace[0].data["location_hint"] == "on the table"
