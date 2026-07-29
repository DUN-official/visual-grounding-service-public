import json

from PIL import Image

from grounding.backends.gpt_guided_owlvit_backend import GPTGuidedOWLViTBackend
from grounding.schemas import (
    BBoxXYXY,
    GroundingCandidate,
    GroundingRequest,
    ImagePayload,
    PerformanceMode,
)


class _Health:
    loaded = True


def _candidate(candidate_id, box, confidence=0.7):
    return GroundingCandidate(
        candidate_id=candidate_id,
        bbox_xyxy=BBoxXYXY(
            x_min=box[0],
            y_min=box[1],
            x_max=box[2],
            y_max=box[3],
        ),
        confidence=confidence,
        label="package",
        source="owlvit",
    )


class _Owl:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def health(self):
        return _Health()

    def detect_candidates(self, request, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class _Response:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class _Responses:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Response(self.payloads.pop(0))


class _Client:
    def __init__(self, payloads):
        self.responses = _Responses(payloads)


def _request(image_path):
    return GroundingRequest(
        image=ImagePayload(path=str(image_path)),
        instruction="find the package beside the chair",
        target_object="package",
        target_phrase="package",
        performance_mode=PerformanceMode.QUALITY,
        maximum_latency_ms=300_000,
    )


def test_quality_mode_runs_selection_refinement_and_edge_review(tmp_path):
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (200, 120), "white").save(image_path)
    owl = _Owl(
        [
            [
                _candidate("owl_1", [20, 20, 80, 90]),
                _candidate("owl_2", [100, 20, 160, 90]),
            ],
            [
                _candidate("local_1", [5, 5, 55, 65]),
                _candidate("local_2", [8, 8, 58, 68]),
            ],
        ]
    )
    backend = GPTGuidedOWLViTBackend(owlvit_backend=owl, openai_model="test")
    backend._started = True
    backend._client = _Client(
        [
            {
                "status": "selected",
                "selected_candidate": 1,
                "confidence": 0.9,
                "reason": "initial",
            },
            {
                "status": "selected",
                "selected_candidate": 2,
                "confidence": 0.92,
                "reason": "refined",
            },
            {"decision": "accept", "confidence": 0.95, "reason": "complete box"},
        ]
    )

    result = backend._ground_impl(_request(image_path))

    assert result.status == "success"
    assert result.predictions[0].candidate_id == "refined_2"
    assert result.metadata["performance_profile"] == "quality"
    assert result.metadata["gpt_request_count"] == 3
    assert backend._client.responses.calls == 3
    assert len(owl.calls) == 2
    assert all(call["use_original_size"] is True for call in owl.calls)
    assert owl.calls[0]["thresholds"][-1] == 0.0


def test_quality_mode_can_recover_from_a_coarse_region(tmp_path):
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (200, 120), "white").save(image_path)
    owl = _Owl([[], [_candidate("local_1", [5, 5, 55, 65])]])
    backend = GPTGuidedOWLViTBackend(owlvit_backend=owl, openai_model="test")
    backend._started = True
    backend._client = _Client(
        [
            {
                "status": "refine",
                "region": [0.2, 0.2, 0.7, 0.8],
                "confidence": 0.8,
                "reason": "coarse",
            },
            {
                "status": "selected",
                "selected_candidate": 1,
                "confidence": 0.9,
                "reason": "refined",
            },
            {"decision": "accept", "confidence": 0.95, "reason": "complete box"},
        ]
    )

    result = backend._ground_impl(_request(image_path))

    assert result.status == "success"
    assert result.predictions[0].candidate_id == "refined_1"
    assert result.metadata["initial_candidate_count"] == 0
    assert result.metadata["refined_candidate_count"] == 1
    assert backend._client.responses.calls == 3
