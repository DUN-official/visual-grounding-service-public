from grounding.backends.gpt_guided_owlvit_backend import GPTGuidedOWLViTBackend
from grounding.schemas import BBoxXYXY


class Owl:
    pass


def backend():
    return GPTGuidedOWLViTBackend(
        owlvit_backend=Owl(),
        openai_model="test",
        minimum_adjustment_confidence=0.0,
        minimum_adjusted_area_ratio=0.8,
        maximum_adjusted_area_ratio=1.6,
        default_box_padding=0.04,
        maximum_edge_adjustment=0.2,
        maximum_edge_contraction=0.06,
    )


def test_extreme_contraction_cannot_create_tiny_box():
    original = BBoxXYXY(x_min=100, y_min=100, x_max=300, y_max=300)
    final, applied, gate = backend()._apply_safe_adjustment(
        original,
        {
            "confidence": 1.0,
            "left_shift": 0.2,
            "top_shift": 0.2,
            "right_shift": -0.2,
            "bottom_shift": -0.2,
        },
        500,
        500,
    )
    assert final.area >= original.area * 0.8
    assert final.x_min <= 112
    assert final.x_max >= 288
