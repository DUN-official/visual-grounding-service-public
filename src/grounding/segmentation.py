from __future__ import annotations

"""Lightweight box-prompted segmentation utilities.

This module deliberately uses OpenCV GrabCut so segmentation can be added with
minimal disruption to the existing grounding pipeline. The existing backends
still produce the box; segmentation is an optional post-processing step.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import BBoxXYXY


@dataclass(slots=True)
class SegmentationOutput:
    mask: np.ndarray
    payload: dict[str, Any]


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('segmentation requires OpenCV (cv2)') from exc
    return cv2


def _clip_box(box: BBoxXYXY | tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    if isinstance(box, BBoxXYXY):
        x1, y1, x2, y2 = box.as_list()
    else:
        x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(float(x1)))))
    y1 = max(0, min(height - 1, int(round(float(y1)))))
    x2 = max(x1 + 1, min(width, int(round(float(x2)))))
    y2 = max(y1 + 1, min(height, int(round(float(y2)))))
    return x1, y1, x2, y2


def _rect_fallback(width: int, height: int, box: tuple[int, int, int, int]) -> SegmentationOutput:
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    payload = {
        'method': 'bbox_fallback',
        'area_pixels': int((y2 - y1) * (x2 - x1)),
        'tight_bbox_xyxy': [x1, y1, x2, y2],
        'contours': [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
    }
    return SegmentationOutput(mask=mask, payload=payload)


def _serialize_mask(mask: np.ndarray, *, max_contours: int = 4) -> dict[str, Any]:
    cv2 = _cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'method': 'grabcut_box_prompt', 'area_pixels': 0, 'tight_bbox_xyxy': None, 'contours': []}

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
    serialized: list[list[list[int]]] = []
    all_points: list[tuple[int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 9.0:
            continue
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(points) >= 3:
            serialized.append(points)
            all_points.extend((pt[0], pt[1]) for pt in points)
    if not serialized:
        return {'method': 'grabcut_box_prompt', 'area_pixels': int(mask.sum() / 255), 'tight_bbox_xyxy': None, 'contours': []}

    xs = [pt[0] for pt in all_points]
    ys = [pt[1] for pt in all_points]
    return {
        'method': 'grabcut_box_prompt',
        'area_pixels': int(mask.sum() / 255),
        'tight_bbox_xyxy': [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
        'contours': serialized,
    }


def segment_from_box(image_rgb: np.ndarray, box: BBoxXYXY | tuple[float, float, float, float], *, pad_fraction: float = 0.04, iterations: int = 2) -> SegmentationOutput:
    cv2 = _cv2()
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError('image is empty')
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 2 or box_h < 2:
        return _rect_fallback(width, height, (x1, y1, x2, y2))

    pad_x = max(1, int(round(box_w * float(pad_fraction))))
    pad_y = max(1, int(round(box_h * float(pad_fraction))))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)
    rect = (int(rx1), int(ry1), int(max(1, rx2 - rx1)), int(max(1, ry2 - ry1)))

    # GrabCut expects BGR.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, int(max(1, iterations)), cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        if int(binary.sum()) <= 0:
            return _rect_fallback(width, height, (x1, y1, x2, y2))
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        if int(binary.sum()) <= 0:
            return _rect_fallback(width, height, (x1, y1, x2, y2))
        return SegmentationOutput(mask=binary, payload=_serialize_mask(binary))
    except Exception:
        return _rect_fallback(width, height, (x1, y1, x2, y2))


def draw_segmentation_overlay(frame_bgr: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.28) -> np.ndarray:
    cv2 = _cv2()
    if mask is None or mask.size == 0:
        return frame_bgr
    overlay = frame_bgr.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, float(alpha), frame_bgr, 1.0 - float(alpha), 0.0)
    frame_bgr[:] = blended
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(frame_bgr, contours, -1, color, 2)
    return frame_bgr
