from PIL import Image

from grounding.backends.gpt_guided_owlvit_backend import GPTGuidedOWLViTBackend
from grounding.schemas import (
    BBoxXYXY, GroundingCandidate, GroundingRequest, ImagePayload,
    PerformanceMode, QuantityIntent,
)


class Health:
    loaded = True


class FakeOwl:
    def health(self): return Health()
    def detect_candidates(self, request, **kwargs):
        return [
            GroundingCandidate(
                candidate_id="owl_1",
                bbox_xyxy=BBoxXYXY(x_min=10, y_min=10, x_max=60, y_max=90),
                confidence=0.7,
                label="person",
                source="owlvit",
            ),
            GroundingCandidate(
                candidate_id="owl_2",
                bbox_xyxy=BBoxXYXY(x_min=70, y_min=10, x_max=120, y_max=90),
                confidence=0.69,
                label="person",
                source="owlvit",
            ),
        ]


class FakeResponse:
    output_text = '''{"status":"selected","selected_candidates":[{"candidate":2,"confidence":0.9,"relation_match":true,"left_shift":0,"top_shift":0,"right_shift":0,"bottom_shift":0}],"reason":"green person"}'''


class FakeResponses:
    def __init__(self): self.calls = 0
    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse()


class FakeClient:
    def __init__(self): self.responses = FakeResponses()


def test_balanced_guided_path_uses_one_gpt_call(tmp_path):
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (150, 100), "white").save(image_path)
    backend = GPTGuidedOWLViTBackend(
        owlvit_backend=FakeOwl(),
        openai_model="test",
        debug_output_dir=None,
    )
    backend._started = True
    backend._client = FakeClient()
    request = GroundingRequest(
        image=ImagePayload(path=str(image_path)),
        instruction="find the man in green",
        target_object="man",
        target_phrase="man",
        quantity=QuantityIntent.ONE,
        attributes=["green"],
        requires_guided_reasoning=True,
        performance_mode=PerformanceMode.BALANCED,
    )
    result = backend._ground_impl(request)
    assert result.status == "success"
    assert result.predictions[0].candidate_id == "owl_2"
    assert backend._client.responses.calls == 1
    assert result.metadata["gpt_request_count"] == 1
