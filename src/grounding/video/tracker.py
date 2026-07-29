"""Precision-oriented OpenCV tracker wrapper."""

from __future__ import annotations

import math


Box = tuple[float, float, float, float]


class OpenCVBoxTracker:
    """Track one target with CSRT-first selection and conservative gates.

    The tracker follows a padded context region but converts each update back to
    the original target-sized box. This is more stable for small objects while
    avoiding visibly oversized output boxes.
    """

    def __init__(
        self,
        tracker_type: str = "AUTO",
        *,
        context_padding: float = 0.18,
        maximum_center_jump: float = 0.30,
        minimum_area_ratio: float = 0.45,
        maximum_area_ratio: float = 2.40,
        template_recovery_threshold: float = 0.58,
        minimum_appearance_similarity: float = 0.55,
    ) -> None:
        self.requested_type = (tracker_type or "AUTO").upper()
        self.active_type: str | None = None
        self.context_padding = max(0.0, min(0.75, float(context_padding)))
        self.maximum_center_jump = max(0.05, min(1.0, float(maximum_center_jump)))
        self.minimum_area_ratio = max(0.05, float(minimum_area_ratio))
        self.maximum_area_ratio = max(self.minimum_area_ratio, float(maximum_area_ratio))
        self.template_recovery_threshold = max(0.0, min(1.0, float(template_recovery_threshold)))
        self.minimum_appearance_similarity = max(
            0.0,
            min(1.0, float(minimum_appearance_similarity)),
        )
        self._tracker = None
        self._previous_tracking_box: Box | None = None
        self._target_fractions: tuple[float, float, float, float] | None = None
        self._template = None
        self.last_quality = 0.0
        self.last_appearance_similarity = 0.0
        self.last_source = "none"

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("video support requires OpenCV") from exc
        return cv2

    def _create(self):
        cv2 = self._cv2()
        preferred = [] if self.requested_type == "AUTO" else [self.requested_type]
        names = [*preferred, "CSRT", "KCF", "MIL"]
        for name in dict.fromkeys(names):
            constructor_name = f"Tracker{name}_create"
            constructor = getattr(cv2, constructor_name, None)
            if constructor is None and hasattr(cv2, "legacy"):
                constructor = getattr(cv2.legacy, constructor_name, None)
            if constructor is not None:
                self.active_type = name
                return constructor()
        raise RuntimeError("no supported OpenCV tracker was found; install opencv-contrib-python")

    def reset(self) -> None:
        self._tracker = None
        self.active_type = None
        self._previous_tracking_box = None
        self._target_fractions = None
        self._template = None
        self.last_quality = 0.0
        self.last_appearance_similarity = 0.0
        self.last_source = "none"

    def initialize(self, frame, bbox_xyxy: Box) -> None:
        target = self._clip(bbox_xyxy, frame.shape[1], frame.shape[0])
        tracking = self._pad(target, frame.shape[1], frame.shape[0], self.context_padding)
        tx1, ty1, tx2, ty2 = target
        px1, py1, px2, py2 = tracking
        pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
        self._target_fractions = (
            (tx1 - px1) / pw,
            (ty1 - py1) / ph,
            (tx2 - px1) / pw,
            (ty2 - py1) / ph,
        )

        tracker = self._create()
        initialized = tracker.init(frame, self._xyxy_to_xywh(tracking))
        if initialized is False:
            raise RuntimeError("OpenCV tracker initialization failed")
        self._tracker = tracker
        self._previous_tracking_box = tracking
        self._template = self._crop_gray(frame, tracking)
        self.last_quality = 1.0
        self.last_appearance_similarity = 1.0
        self.last_source = "grounding"

    def update(self, frame) -> tuple[bool, Box | None]:
        if self._tracker is None or self._previous_tracking_box is None:
            return False, None

        ok, xywh = self._tracker.update(frame)
        if ok:
            tracking = self._clip(self._xywh_to_xyxy(xywh), frame.shape[1], frame.shape[0])
            if self._plausible(self._previous_tracking_box, tracking, frame.shape[1], frame.shape[0]):
                motion_quality = self._motion_quality(
                    self._previous_tracking_box,
                    tracking,
                    frame.shape[1],
                    frame.shape[0],
                )
                appearance = self._appearance_similarity(frame, tracking)
                self.last_appearance_similarity = appearance
                if appearance >= self.minimum_appearance_similarity:
                    self._previous_tracking_box = tracking
                    self.last_quality = min(motion_quality, appearance)
                    self.last_source = "tracker"
                    return True, self._target_from_tracking(tracking)

        recovered = self._template_recover(frame)
        if recovered is not None:
            self._previous_tracking_box = recovered
            self.last_source = "template_recovery"
            self.last_quality = max(self.last_quality, self.template_recovery_threshold)
            self.last_appearance_similarity = self.last_quality
            try:
                tracker = self._create()
                tracker.init(frame, self._xyxy_to_xywh(recovered))
                self._tracker = tracker
            except Exception:
                pass
            return True, self._target_from_tracking(recovered)

        self.last_quality = 0.0
        self.last_appearance_similarity = 0.0
        self.last_source = "lost"
        return False, None

    def _appearance_similarity(self, frame, tracking: Box) -> float:
        cv2 = self._cv2()
        if self._template is None or self._template.size == 0:
            return 1.0
        candidate = self._crop_gray(frame, tracking)
        if candidate is None or candidate.size == 0:
            return 0.0
        if candidate.shape != self._template.shape:
            candidate = cv2.resize(
                candidate,
                (self._template.shape[1], self._template.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        result = cv2.matchTemplate(
            candidate,
            self._template,
            cv2.TM_CCOEFF_NORMED,
        )
        score = float(result[0, 0])
        if not math.isfinite(score):
            difference = cv2.absdiff(candidate, self._template)
            score = 1.0 - float(difference.mean()) / 255.0
        return max(0.0, min(1.0, score))

    def _target_from_tracking(self, tracking: Box) -> Box:
        if self._target_fractions is None:
            return tracking
        x1, y1, x2, y2 = tracking
        width, height = x2 - x1, y2 - y1
        fx1, fy1, fx2, fy2 = self._target_fractions
        return (
            x1 + fx1 * width,
            y1 + fy1 * height,
            x1 + fx2 * width,
            y1 + fy2 * height,
        )

    def _template_recover(self, frame) -> Box | None:
        cv2 = self._cv2()
        if self._template is None or self._previous_tracking_box is None:
            return None
        x1, y1, x2, y2 = self._previous_tracking_box
        width, height = x2 - x1, y2 - y1
        search = self._clip(
            (x1 - width, y1 - height, x2 + width, y2 + height),
            frame.shape[1],
            frame.shape[0],
        )
        sx1, sy1, sx2, sy2 = [int(round(v)) for v in search]
        gray = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
        template = self._template
        if gray.size == 0 or template.size == 0:
            return None
        if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
            return None
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if float(score) < self.template_recovery_threshold:
            return None
        tx, ty = location
        recovered = (
            float(sx1 + tx),
            float(sy1 + ty),
            float(sx1 + tx + template.shape[1]),
            float(sy1 + ty + template.shape[0]),
        )
        if not self._plausible(self._previous_tracking_box, recovered, frame.shape[1], frame.shape[0]):
            return None
        self.last_quality = float(score)
        return recovered

    def _plausible(self, previous: Box, current: Box, frame_width: int, frame_height: int) -> bool:
        previous_area = self._area(previous)
        current_area = self._area(current)
        if previous_area <= 1.0 or current_area <= 1.0:
            return False
        ratio = current_area / previous_area
        if ratio < self.minimum_area_ratio or ratio > self.maximum_area_ratio:
            return False
        px, py = self._center(previous)
        cx, cy = self._center(current)
        jump = math.hypot(cx - px, cy - py)
        diagonal = max(1.0, math.hypot(frame_width, frame_height))
        return jump / diagonal <= self.maximum_center_jump

    @staticmethod
    def _motion_quality(previous: Box, current: Box, frame_width: int, frame_height: int) -> float:
        px, py = OpenCVBoxTracker._center(previous)
        cx, cy = OpenCVBoxTracker._center(current)
        jump = math.hypot(cx - px, cy - py) / max(1.0, math.hypot(frame_width, frame_height))
        area_ratio = OpenCVBoxTracker._area(current) / max(1.0, OpenCVBoxTracker._area(previous))
        scale_penalty = min(1.0, abs(math.log(max(area_ratio, 1e-6))) / math.log(2.5))
        return max(0.0, min(1.0, 1.0 - 1.7 * jump - 0.5 * scale_penalty))

    @staticmethod
    def _pad(box: Box, frame_width: int, frame_height: int, fraction: float) -> Box:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        return OpenCVBoxTracker._clip(
            (x1 - width * fraction, y1 - height * fraction, x2 + width * fraction, y2 + height * fraction),
            frame_width,
            frame_height,
        )

    @staticmethod
    def _clip(box: Box, frame_width: int, frame_height: int) -> Box:
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(frame_width - 1), float(x1)))
        y1 = max(0.0, min(float(frame_height - 1), float(y1)))
        x2 = max(x1 + 1.0, min(float(frame_width), float(x2)))
        y2 = max(y1 + 1.0, min(float(frame_height), float(y2)))
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_xywh(box: Box) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return int(round(x1)), int(round(y1)), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))

    @staticmethod
    def _xywh_to_xyxy(box) -> Box:
        x, y, width, height = [float(value) for value in box]
        return x, y, x + width, y + height

    @staticmethod
    def _crop_gray(frame, box: Box):
        cv2 = OpenCVBoxTracker._cv2()
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        crop = frame[y1:y2, x1:x2]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None

    @staticmethod
    def _area(box: Box) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    @staticmethod
    def _center(box: Box) -> tuple[float, float]:
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    def _context_box(
        self,
        target: Box,
        frame_width: int,
        frame_height: int,
    ) -> Box:
        return self._pad(target, frame_width, frame_height, self.context_padding)

    @staticmethod
    def _fraction_inside(target: Box, context: Box) -> tuple[float, float, float, float]:
        tx1, ty1, tx2, ty2 = target
        cx1, cy1, cx2, cy2 = context
        width = max(1.0, cx2 - cx1)
        height = max(1.0, cy2 - cy1)
        return (
            (tx1 - cx1) / width,
            (ty1 - cy1) / height,
            (tx2 - cx1) / width,
            (ty2 - cy1) / height,
        )

    @classmethod
    def _target_from_context(
        cls,
        context: Box,
        fractions: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Box:
        x1, y1, x2, y2 = context
        width = x2 - x1
        height = y2 - y1
        fx1, fy1, fx2, fy2 = fractions
        target = (
            x1 + fx1 * width,
            y1 + fy1 * height,
            x1 + fx2 * width,
            y1 + fy2 * height,
        )
        return cls._clip(target, frame_width, frame_height)

    def _geometry_is_valid(self, previous: Box, current: Box) -> bool:
        frame_width = max(1, int(max(previous[2], current[2])) + 1)
        frame_height = max(1, int(max(previous[3], current[3])) + 1)
        return self._plausible(previous, current, frame_width, frame_height)


__all__ = ["Box", "OpenCVBoxTracker"]
