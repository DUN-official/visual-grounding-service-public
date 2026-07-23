from grounding.router import GroundingRouter, RoutingPolicy
from grounding.schemas import BackendHealth, GroundingRequest, HealthStatus, ImagePayload

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
