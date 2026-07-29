"""Adapters that expose image grounding to video processors."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ..llm_task_parser import LLMVideoTaskParser
from ..openai_context import use_openai_api_key
from ..schemas import GroundingRequest, ImagePayload
from ..task_parser import normalize_text


@dataclass(slots=True)
class VideoGroundingPlan:
    instruction: str
    grounding_instruction: str
    target_object: str | None
    target_phrase: str | None
    attributes: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    anchor_objects: list[str] = field(default_factory=list)
    anchor_phrases: list[str] = field(default_factory=list)
    requested_backend: str | None = None
    performance_mode: str = "balanced"


@dataclass(slots=True)
class GroundingObservation:
    success: bool
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float
    backend: str | None
    status: str
    message: str
    selected_label: str | None
    raw: dict[str, Any]

    @property
    def label(self) -> str | None:
        return self.selected_label


def build_video_grounding_plan(
    instruction: str,
    *,
    performance_mode: str = "balanced",
    preferred_backend: str | None = None,
    parser_mode: str = "local",
    structured=None,
) -> VideoGroundingPlan:
    """Build a target-safe grounding instruction for video acquisition."""

    if structured is None:
        structured = LLMVideoTaskParser().parse(
            instruction,
            use_llm=parser_mode.lower() == "llm",
        )

    explicit_backend = normalize_text(preferred_backend)
    if explicit_backend in {"", "auto", "none"}:
        explicit_backend = ""
    recommended = getattr(structured, "recommended_backend", None)
    guided = bool(getattr(structured, "requires_guided_reasoning", False))
    requested_backend = explicit_backend or recommended
    if not requested_backend and guided:
        requested_backend = "gpt_guided_owlvit"

    target_object = getattr(structured, "target_object", None)
    target_phrase = getattr(structured, "target_phrase", None) or target_object
    anchor_objects = list(getattr(structured, "anchor_objects", []) or [])
    anchor_phrases = list(getattr(structured, "anchor_phrases", []) or anchor_objects)

    lines = [instruction.strip()]
    if target_phrase:
        lines.append(f"Target to select: {target_phrase}.")
    if anchor_phrases:
        lines.append(
            "Reference objects, never select these as the target: "
            + ", ".join(anchor_phrases)
            + "."
        )

    return VideoGroundingPlan(
        instruction=instruction,
        grounding_instruction=" ".join(lines),
        target_object=target_object,
        target_phrase=target_phrase,
        attributes=list(getattr(structured, "attributes", []) or []),
        relations=list(getattr(structured, "relations", []) or []),
        anchor_objects=anchor_objects,
        anchor_phrases=anchor_phrases,
        requested_backend=requested_backend or None,
        performance_mode=performance_mode,
    )


def _encode_frame(frame, jpeg_quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video support requires OpenCV") from exc

    quality = max(40, min(100, int(jpeg_quality)))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the video frame as JPEG")
    return encoded.tobytes()


def _bbox_from_value(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return (
            float(value["x_min"]),
            float(value["y_min"]),
            float(value["x_max"]),
            float(value["y_max"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _label_matches(label: str | None, expected: str | None) -> bool:
    normalized_label = normalize_text(label)
    normalized_expected = normalize_text(expected)
    if not normalized_label or not normalized_expected:
        return False
    return (
        normalized_label == normalized_expected
        or normalized_expected in normalized_label
        or normalized_label in normalized_expected
    )


def _observation_from_payload(
    payload: dict[str, Any],
    plan: VideoGroundingPlan | None = None,
) -> GroundingObservation:
    predictions = [
        item for item in (payload.get("predictions") or []) if isinstance(item, dict)
    ]
    selected = None
    anchor_only = False

    if plan is not None and predictions:
        target_matches = [
            item
            for item in predictions
            if _label_matches(item.get("label"), plan.target_object)
            or _label_matches(item.get("label"), plan.target_phrase)
        ]
        if target_matches:
            selected = max(
                target_matches,
                key=lambda item: float(item.get("confidence") or 0.0),
            )
        else:
            anchor_only = all(
                any(_label_matches(item.get("label"), anchor) for anchor in plan.anchor_objects)
                for item in predictions
            )

    if selected is None and predictions and not anchor_only:
        selected = predictions[0]

    if selected is not None:
        bbox = _bbox_from_value(selected.get("bbox_xyxy"))
        label = selected.get("label")
        confidence = float(selected.get("confidence") or payload.get("confidence") or 0.0)
    else:
        bbox = None if anchor_only else _bbox_from_value(payload.get("bbox_xyxy"))
        label = None
        confidence = float(payload.get("confidence") or 0.0)

    if not label:
        candidates = payload.get("candidates") or []
        if candidates and isinstance(candidates[0], dict):
            label = candidates[0].get("label")

    status = str(payload.get("status", "failed"))
    message = str(payload.get("error") or "")
    if anchor_only:
        status = "failed"
        message = "anchor-only result rejected; target acquisition required"
    elif not message:
        trace = payload.get("trace") or []
        if trace and isinstance(trace[-1], dict):
            message = str(trace[-1].get("message") or "")

    return GroundingObservation(
        success=status == "success" and bbox is not None,
        bbox_xyxy=bbox,
        confidence=confidence,
        backend=payload.get("backend_used"),
        status=status,
        message=message,
        selected_label=str(label) if label else None,
        raw=payload,
    )


class HTTPGroundingServiceAdapter:
    """Send selected video frames through the FastAPI image endpoint."""

    def __init__(self, service_url: str = "http://127.0.0.1:8000") -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("video support requires the video dependency group") from exc
        self.service_url = service_url.rstrip("/")
        self._client = httpx.Client(timeout=None)

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        response = self._client.get(f"{self.service_url}/health")
        response.raise_for_status()
        return response.json()

    def ground_frame(
        self,
        frame,
        *,
        instruction: str,
        target_object: str | None = None,
        location_hint: str | None = None,
        action: str | None = None,
        performance_mode: str = "balanced",
        preferred_backend: str | None = None,
        maximum_latency_ms: int = 200_000,
        jpeg_quality: int = 90,
    ) -> GroundingObservation:
        image_bytes = _encode_frame(frame, jpeg_quality)
        data: dict[str, str] = {
            "instruction": instruction,
            "performance_mode": performance_mode,
            "parser_mode": "local",
            "maximum_results": "1",
            "maximum_latency_ms": str(int(maximum_latency_ms)),
        }
        for key, value in {
            "target_object": target_object,
            "location_hint": location_hint,
            "action": action,
            "preferred_backend": preferred_backend,
        }.items():
            if value:
                data[key] = str(value)

        response = self._client.post(
            f"{self.service_url}/v1/ground/upload",
            data=data,
            files={"image": ("video_frame.jpg", image_bytes, "image/jpeg")},
        )
        response.raise_for_status()
        return _observation_from_payload(response.json())


class LocalGroundingServiceAdapter:
    """Run selected video frames through an in-process grounding service."""

    def __init__(
        self,
        service,
        *,
        lock: RLock | None = None,
        api_key: str | None = None,
        prepare_backend=None,
    ) -> None:
        self.service = service
        self._lock = lock or RLock()
        self.api_key = api_key
        self._prepare_backend = prepare_backend

    def close(self) -> None:
        return None

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                name: state.model_dump(mode="json")
                for name, state in self.service.health().items()
            }

    def ground_frame(
        self,
        frame,
        *,
        instruction: str,
        target_object: str | None = None,
        location_hint: str | None = None,
        action: str | None = None,
        performance_mode: str = "balanced",
        preferred_backend: str | None = None,
        maximum_latency_ms: int = 200_000,
        jpeg_quality: int = 90,
    ) -> GroundingObservation:
        image_bytes = _encode_frame(frame, jpeg_quality)
        request = GroundingRequest(
            image=ImagePayload(
                base64_data=base64.b64encode(image_bytes).decode("ascii"),
                media_type="image/jpeg",
            ),
            instruction=instruction,
            target_object=target_object,
            location_hint=location_hint,
            action=action,
            performance_mode=performance_mode,
            preferred_backend=preferred_backend,
            maximum_results=1,
            maximum_latency_ms=maximum_latency_ms,
            metadata={"input_mode": "video_frame", "parser_mode": "local"},
        )
        if self._prepare_backend is not None:
            self._prepare_backend(
                preferred_backend,
                instruction,
                api_key=self.api_key,
                performance_mode=performance_mode,
            )
        with self._lock, use_openai_api_key(self.api_key):
            result = self.service.ground(request)
        return _observation_from_payload(result.model_dump(mode="json"))
