from ..schemas import BBoxXYXY

def box_iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
    left = max(a.x_min, b.x_min)
    top = max(a.y_min, b.y_min)
    right = min(a.x_max, b.x_max)
    bottom = min(a.y_max, b.y_max)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.area + b.area - intersection
    return 0.0 if union <= 0 else float(intersection / union)

def quality_grade(iou, strict_threshold=0.50, success_threshold=0.25, weak_threshold=0.10):
    if iou >= strict_threshold:
        return "strict_success"
    if iou >= success_threshold:
        return "success"
    if iou >= weak_threshold:
        return "weak_overlap"
    return "poor"
