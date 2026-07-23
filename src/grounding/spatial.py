"""Fast image-space relation scoring for common grounding constraints."""

from __future__ import annotations

import math

from .schemas import GroundingCandidate, SpatialConstraint
from .task_parser import normalize_text

GROUND_ANCHORS = {"floor", "ground"}
CEILING_ANCHORS = {"ceiling"}


def rank_candidates_by_constraints(candidates, constraints, anchors_by_name, image_size):
    ranked = []
    for candidate in candidates:
        relation_scores = []
        for constraint in constraints:
            anchors = anchors_by_name.get(normalize_text(constraint.anchor), [])
            relation_scores.append(
                score_constraint(candidate, constraint, anchors, image_size)
            )
        relation_score = sum(relation_scores) / len(relation_scores) if relation_scores else 0.0
        combined = 0.35 * candidate.confidence + 0.65 * relation_score
        ranked.append((combined, relation_score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def score_constraint(target, constraint: SpatialConstraint, anchors, image_size):
    relation = normalize_text(constraint.relation)
    anchor_name = normalize_text(constraint.anchor)
    width, height = image_size
    box = target.bbox_xyxy
    tx, ty = box.center

    if anchor_name in GROUND_ANCHORS:
        bottom = box.y_max / max(1.0, height)
        center = ty / max(1.0, height)
        return _clamp(0.65 * bottom + 0.35 * center)
    if anchor_name in CEILING_ANCHORS:
        return _clamp(1.0 - box.y_min / max(1.0, height))
    if not anchors:
        return 0.0

    scores = [_score_pair(box, anchor.bbox_xyxy, relation, width, height) for anchor in anchors]
    if relation == "furthest_from":
        return max(scores)
    return max(scores)


def _score_pair(target, anchor, relation, image_width, image_height):
    tx, ty = target.center
    ax, ay = anchor.center
    diagonal = max(1.0, math.hypot(image_width, image_height))
    distance = math.hypot(tx - ax, ty - ay) / diagonal
    proximity = _clamp(1.0 - distance * 3.0)
    horizontal_overlap = _axis_overlap(target.x_min, target.x_max, anchor.x_min, anchor.x_max)
    vertical_overlap = _axis_overlap(target.y_min, target.y_max, anchor.y_min, anchor.y_max)

    if relation in {"beside", "near", "closest_to", "opposite"}:
        return proximity
    if relation == "furthest_from":
        return _clamp(distance * 2.0)
    if relation == "left_of":
        return _clamp((ax - tx) / max(1.0, image_width) * 4.0)
    if relation == "right_of":
        return _clamp((tx - ax) / max(1.0, image_width) * 4.0)
    if relation == "above":
        return _clamp((ay - ty) / max(1.0, image_height) * 4.0)
    if relation == "below":
        return _clamp((ty - ay) / max(1.0, image_height) * 4.0)
    if relation == "inside":
        inside_x = target.x_min >= anchor.x_min and target.x_max <= anchor.x_max
        inside_y = target.y_min >= anchor.y_min and target.y_max <= anchor.y_max
        return 1.0 if inside_x and inside_y else 0.0
    if relation == "outside":
        inside_x = target.x_min >= anchor.x_min and target.x_max <= anchor.x_max
        inside_y = target.y_min >= anchor.y_min and target.y_max <= anchor.y_max
        return 0.0 if inside_x and inside_y else proximity
    if relation == "on":
        vertical_gap = abs(target.y_max - anchor.y_min) / max(1.0, image_height)
        overlap_score = horizontal_overlap
        # Objects on furniture can overlap the furniture detector box.
        return _clamp(0.55 * overlap_score + 0.45 * (1.0 - min(1.0, vertical_gap * 5.0)))
    if relation in {"in_front_of", "behind", "at_end_of", "between"}:
        return 0.5 * proximity + 0.25 * horizontal_overlap + 0.25 * vertical_overlap
    return 0.0


def clear_winner(ranked, minimum_score=0.72, minimum_margin=0.12):
    if not ranked:
        return False
    top = ranked[0][0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    return top >= minimum_score and (top - second) >= minimum_margin


def _axis_overlap(a0, a1, b0, b1):
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    denominator = max(1.0, min(a1 - a0, b1 - b0))
    return _clamp(intersection / denominator)


def _clamp(value):
    return max(0.0, min(1.0, float(value)))
