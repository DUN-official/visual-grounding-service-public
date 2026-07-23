from grounding.evaluation.metrics import box_iou, quality_grade
from grounding.schemas import BBoxXYXY

def test_iou_identity():
    box = BBoxXYXY(x_min=0, y_min=0, x_max=10, y_max=10)
    assert box_iou(box, box) == 1.0

def test_quality_thresholds():
    assert quality_grade(0.60) == "strict_success"
    assert quality_grade(0.30) == "success"
    assert quality_grade(0.15) == "weak_overlap"
    assert quality_grade(0.05) == "poor"
