from types import SimpleNamespace

from grounding.llm_task_parser import parse_task_with_service
from grounding.video.service_adapter import build_video_grounding_plan


class _Response:
    def __init__(self, text):
        self.output_text = text


class _Responses:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Response(self.text)


class _Client:
    def __init__(self, text):
        self.responses = _Responses(text)


class _Backend:
    openai_model = "test-model"

    def __init__(self, text, loaded=True):
        self._client = _Client(text) if loaded else None
        self._loaded = loaded

    def health(self):
        return SimpleNamespace(loaded=self._loaded)


class _Service:
    def __init__(self, backend):
        self.backends = {"gpt_guided_owlvit": backend} if backend else {}


def _valid_payload(target="toy car"):
    return f'''{{
      "action": "track",
      "target": {{
        "object": "{target}",
        "phrase": "{target}",
        "attributes": [],
        "quantity": "one",
        "minimum_count": 1
      }},
      "relations": [{{
        "type": "beside",
        "anchor_object": "water bottle",
        "anchor_phrase": "green water bottle",
        "anchor_attributes": ["green"],
        "raw_text": "beside the green water bottle"
      }}],
      "requires_instance_selection": true,
      "reasoning_complexity": "multi_constraint",
      "recommended_backend": "gpt_guided_owlvit",
      "confidence": 0.96
    }}'''


def test_llm_parser_separates_target_and_anchor():
    backend = _Backend(_valid_payload())
    parsed = parse_task_with_service(
        _Service(backend),
        "track toy car beside green water bottle",
        parser_mode="llm",
    )
    assert parsed.parser_source == "llm"
    assert parsed.target_object == "toy car"
    assert parsed.anchor_objects == ["water bottle"]
    assert parsed.anchor_phrases == ["green water bottle"]
    assert parsed.recommended_backend == "gpt_guided_owlvit"
    assert backend._client.responses.calls == 1


def test_cross_check_prevents_anchor_becoming_target():
    parsed = parse_task_with_service(
        _Service(_Backend(_valid_payload(target="water bottle"))),
        "track toy car beside green water bottle",
        parser_mode="llm",
    )
    assert parsed.target_object == "toy car"
    assert parsed.fallback_reason is not None
    assert "reference anchor" in parsed.fallback_reason


def test_llm_unavailable_uses_local_fallback():
    parsed = parse_task_with_service(
        _Service(None),
        "track toy car beside green water bottle",
        parser_mode="llm",
    )
    assert parsed.parser_source == "local_fallback"
    assert parsed.target_object == "toy car"
    assert parsed.anchor_objects == ["green water bottle"]


def test_video_plan_uses_llm_backend_recommendation():
    structured = parse_task_with_service(
        _Service(_Backend(_valid_payload())),
        "track toy car beside green water bottle",
        parser_mode="llm",
    )
    plan = build_video_grounding_plan(
        "track toy car beside green water bottle",
        performance_mode="balanced",
        parser_mode="llm",
        structured=structured,
    )
    assert plan.target_object == "toy car"
    assert plan.anchor_objects == ["water bottle"]
    assert plan.anchor_phrases == ["green water bottle"]
    assert plan.requested_backend == "gpt_guided_owlvit"
    assert "never select these" in plan.grounding_instruction.lower()
