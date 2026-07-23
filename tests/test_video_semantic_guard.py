from grounding.task_parser import parse_grounding_prompt
from grounding.video.service_adapter import (
    _observation_from_payload,
    build_video_grounding_plan,
)


def test_track_action_preserves_target_and_anchor():
    parsed = parse_grounding_prompt("track toy car beside green water bottle")
    assert parsed.action == "find"
    assert parsed.target_object == "toy car"
    assert parsed.anchor_objects == ["green water bottle"]
    assert parsed.requires_guided_reasoning is True


def test_video_auto_backend_uses_gpt_for_guided_prompt():
    plan = build_video_grounding_plan(
        "track toy car beside green water bottle",
        performance_mode="balanced",
    )
    assert plan.target_object == "toy car"
    assert plan.anchor_objects == ["green water bottle"]
    assert plan.requested_backend == "gpt_guided_owlvit"


def test_explicit_backend_is_preserved():
    plan = build_video_grounding_plan(
        "track toy car beside green water bottle",
        preferred_backend="owlvit",
    )
    assert plan.requested_backend == "owlvit"


def test_anchor_only_prediction_is_rejected():
    plan = build_video_grounding_plan("track toy car beside green water bottle")
    payload = {
        "status": "success",
        "backend_used": "gpt_guided_owlvit",
        "bbox_xyxy": {"x_min": 1, "y_min": 2, "x_max": 20, "y_max": 40},
        "confidence": 0.95,
        "predictions": [{
            "bbox_xyxy": {"x_min": 1, "y_min": 2, "x_max": 20, "y_max": 40},
            "confidence": 0.95,
            "label": "water bottle",
        }],
    }
    observation = _observation_from_payload(payload, plan)
    assert observation.success is False
    assert observation.bbox_xyxy is None
    assert "anchor-only" in observation.message


def test_target_prediction_is_selected_over_anchor_prediction():
    plan = build_video_grounding_plan("track toy car beside green water bottle")
    payload = {
        "status": "success",
        "backend_used": "gpt_guided_owlvit",
        "bbox_xyxy": {"x_min": 1, "y_min": 2, "x_max": 20, "y_max": 40},
        "confidence": 0.96,
        "predictions": [
            {
                "bbox_xyxy": {"x_min": 1, "y_min": 2, "x_max": 20, "y_max": 40},
                "confidence": 0.96,
                "label": "green water bottle",
            },
            {
                "bbox_xyxy": {"x_min": 50, "y_min": 60, "x_max": 100, "y_max": 110},
                "confidence": 0.75,
                "label": "toy car",
            },
        ],
    }
    observation = _observation_from_payload(payload, plan)
    assert observation.success is True
    assert observation.selected_label == "toy car"
    assert observation.bbox_xyxy == (50.0, 60.0, 100.0, 110.0)
