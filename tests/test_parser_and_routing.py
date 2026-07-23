from grounding.router import GroundingRouter, RoutingPolicy
from grounding.schemas import (
    BackendHealth, GroundingRequest, HealthStatus, ImagePayload,
    QuantityIntent, ReasoningComplexity,
)
from grounding.task_parser import parse_grounding_prompt


def _health(*names):
    return {
        name: BackendHealth(backend=name, status=HealthStatus.READY, loaded=True)
        for name in names
    }


def test_plural_and_multiple_relations_are_preserved():
    parsed = parse_grounding_prompt(
        "look for the packages on the floor beside the table"
    )
    assert parsed.target_object == "package"
    assert parsed.target_phrase == "packages"
    assert parsed.quantity == QuantityIntent.MULTIPLE
    assert parsed.minimum_count == 2
    assert parsed.maximum_results == 10
    assert [item.relation for item in parsed.relations] == ["on", "beside"]
    assert [item.anchor for item in parsed.relations] == ["floor", "table"]
    assert parsed.reasoning_complexity == ReasoningComplexity.MULTI_CONSTRAINT


def test_attributes_are_not_lost_from_target_context():
    parsed = parse_grounding_prompt("find the small brown package near the door")
    assert parsed.target_object == "package"
    assert parsed.target_phrase == "small brown package"
    assert parsed.attributes == ["small", "brown"]
    assert parsed.requires_guided_reasoning is True


def test_in_color_attribute_routes_to_guided_backend():
    parsed = parse_grounding_prompt("find the man in green")
    request = GroundingRequest(
        image=ImagePayload(path="/tmp/image.jpg"),
        instruction="find the man in green",
        target_object=parsed.target_object,
        target_phrase=parsed.target_phrase,
        attributes=parsed.attributes,
        requires_guided_reasoning=parsed.requires_guided_reasoning,
        reasoning_complexity=parsed.reasoning_complexity,
    )
    router = GroundingRouter(RoutingPolicy(yolo_target_aliases={"person": ["man"]}))
    decision = router.route(request, _health("yolo", "owlvit", "gpt_guided_owlvit"))
    assert decision.selected_backend == "gpt_guided_owlvit"
    assert "visual_attribute" in decision.guidance_reasons


def test_service_can_preserve_plural_intent_but_force_single_result(tmp_path):
    from grounding.evaluation.logger import GroundingJSONLLogger
    from grounding.services.local_service import LocalGroundingService
    from conftest import FakeGroundingBackend

    service = LocalGroundingService(
        backends={"gpt_guided_owlvit": FakeGroundingBackend("gpt_guided_owlvit")},
        router=GroundingRouter(RoutingPolicy(remote_fallback_enabled=False)),
        logger=GroundingJSONLLogger(tmp_path / "log.jsonl"),
    )
    request = GroundingRequest(
        image=ImagePayload(path="/tmp/image.jpg"),
        instruction="find all packages on the floor",
        maximum_results=1,
    )
    effective, _ = service.prepare_request(request)
    assert effective.quantity == QuantityIntent.ALL
    assert effective.minimum_count == 2
    assert effective.maximum_results == 1
