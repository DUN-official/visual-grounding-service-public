from grounding.router import GroundingRouter, RoutingPolicy
from grounding.schemas import (
    BackendHealth,
    GroundingRequest,
    HealthStatus,
    ImagePayload,
    PerformanceMode,
)

def health(*names):
    return {
        name: BackendHealth(backend=name, status=HealthStatus.READY, loaded=True)
        for name in names
    }

def request(instruction, target):
    return GroundingRequest(
        image=ImagePayload(path="/tmp/image.jpg"),
        instruction=instruction,
        target_object=target,
    )

def test_known_target_routes_to_yolo():
    router = GroundingRouter(
        RoutingPolicy(yolo_target_aliases={"person": ["man"]})
    )
    decision = router.route(
        request("find the person", "person"),
        health("yolo", "owlvit", "gpt_guided_owlvit"),
    )
    assert decision.selected_backend == "yolo"

def test_relation_routes_to_gpt_guided():
    router = GroundingRouter(
        RoutingPolicy(yolo_target_aliases={"person": ["man"]})
    )
    decision = router.route(
        request("find the person near the table", "person"),
        health("yolo", "owlvit", "gpt_guided_owlvit"),
    )
    assert decision.selected_backend == "gpt_guided_owlvit"

def test_open_vocabulary_routes_to_owlvit():
    router = GroundingRouter(
        RoutingPolicy(yolo_target_aliases={"person": ["man"]})
    )
    decision = router.route(
        request("find the elevator panel", "elevator panel"),
        health("yolo", "owlvit"),
    )
    assert decision.selected_backend == "owlvit"

def test_remote_fallback():
    router = GroundingRouter(
        RoutingPolicy(yolo_target_aliases={"person": ["man"]})
    )
    decision = router.route(
        request("find the person", "person"),
        health("remote"),
    )
    assert decision.selected_backend == "remote"


def test_quality_simple_target_routes_to_guided_backend():
    router = GroundingRouter(
        RoutingPolicy(yolo_target_aliases={"person": ["man"]})
    )
    quality_request = request("find the person", "person").model_copy(
        update={"performance_mode": PerformanceMode.QUALITY}
    )
    decision = router.route(
        quality_request,
        health("yolo", "owlvit", "gpt_guided_owlvit"),
    )
    assert decision.selected_backend == "gpt_guided_owlvit"


def test_fast_relation_stays_on_local_backend():
    router = GroundingRouter(
        RoutingPolicy(yolo_target_aliases={"person": ["man"]})
    )
    fast_request = request("find the person near the table", "person").model_copy(
        update={"performance_mode": PerformanceMode.FAST}
    )
    decision = router.route(
        fast_request,
        health("yolo", "owlvit", "gpt_guided_owlvit"),
    )
    assert decision.selected_backend == "yolo"
