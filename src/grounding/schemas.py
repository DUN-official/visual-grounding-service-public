from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GroundingStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CLARIFICATION_REQUIRED = "clarification_required"
    INVALID_REQUEST = "invalid_request"


class HealthStatus(StrEnum):
    READY = "ready"
    STARTING = "starting"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


class QuantityIntent(StrEnum):
    ONE = "one"
    MULTIPLE = "multiple"
    ALL = "all"
    UNKNOWN = "unknown"


class ReasoningComplexity(StrEnum):
    SIMPLE = "simple"
    GUIDED = "guided"
    MULTI_CONSTRAINT = "multi_constraint"


class PerformanceMode(StrEnum):
    QUALITY = "quality"
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class BBoxXYXY(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def validate_order(self):
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bbox must satisfy x_max > x_min and y_max > y_min")
        return self

    def as_list(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def clipped(self, width: int, height: int):
        x_min = max(0.0, min(float(width), self.x_min))
        y_min = max(0.0, min(float(height), self.y_min))
        x_max = max(0.0, min(float(width), self.x_max))
        y_max = max(0.0, min(float(height), self.y_max))
        if x_max <= x_min:
            x_min = max(0.0, min(max(0.0, float(width) - 1.0), x_min))
            x_max = min(float(width), x_min + 1.0)
        if y_max <= y_min:
            y_min = max(0.0, min(max(0.0, float(height) - 1.0), y_min))
            y_max = min(float(height), y_min + 1.0)
        return BBoxXYXY(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def padded(self, fraction: float, width: int, height: int):
        return BBoxXYXY(
            x_min=self.x_min - self.width * fraction,
            y_min=self.y_min - self.height * fraction,
            x_max=self.x_max + self.width * fraction,
            y_max=self.y_max + self.height * fraction,
        ).clipped(width, height)


class SpatialConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: str
    anchor: str
    raw_text: str = ""


class ImagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    base64_data: str | None = None
    media_type: str = "image/jpeg"

    @model_validator(mode="after")
    def require_one_source(self):
        if int(bool(self.path)) + int(bool(self.base64_data)) != 1:
            raise ValueError("image must contain exactly one of path or base64_data")
        return self

    @classmethod
    def from_path(cls, path: str | Path):
        return cls(path=str(path))


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    stage: str
    status: str = "ok"
    message: str = ""
    duration_ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class GroundingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    bbox_xyxy: BBoxXYXY
    confidence: float = Field(ge=0.0, le=1.0)
    label: str
    source: str
    relation_match: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundingPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox_xyxy: BBoxXYXY
    confidence: float = Field(ge=0.0, le=1.0)
    label: str | None = None
    relation_match: bool | None = None
    candidate_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    image: ImagePayload
    instruction: str = Field(min_length=1)
    target_object: str | None = None
    target_phrase: str | None = None
    location_hint: str | None = None
    action: str | None = None
    quantity: QuantityIntent = QuantityIntent.UNKNOWN
    minimum_count: int = Field(default=1, ge=1, le=100)
    maximum_results: int = Field(default=1, ge=1, le=100)
    attributes: list[str] = Field(default_factory=list)
    relations: list[SpatialConstraint] = Field(default_factory=list)
    anchor_objects: list[str] = Field(default_factory=list)
    reasoning_complexity: ReasoningComplexity = ReasoningComplexity.SIMPLE
    requires_guided_reasoning: bool = False
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    maximum_latency_ms: int = Field(default=30_000, ge=1, le=300_000)
    preferred_backend: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "instruction", "target_object", "target_phrase", "location_hint", "action",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    status: GroundingStatus
    bbox_xyxy: BBoxXYXY | None = None
    predictions: list[GroundingPrediction] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    relation_match: bool | None = None
    backend_used: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    clarification_required: bool = False
    trace: list[TraceEvent] = Field(default_factory=list)
    candidates: list[GroundingCandidate] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_success(self):
        if self.bbox_xyxy is None and self.predictions:
            self.bbox_xyxy = self.predictions[0].bbox_xyxy
        if self.confidence is None and self.predictions:
            self.confidence = self.predictions[0].confidence
        if self.status == GroundingStatus.SUCCESS and self.bbox_xyxy is None:
            raise ValueError("successful result requires bbox_xyxy or predictions")
        return self

    @classmethod
    def failure(
        cls,
        request,
        *,
        status=GroundingStatus.FAILED,
        backend_used=None,
        message,
        clarification_required=False,
        trace=None,
    ):
        return cls(
            request_id=request.request_id,
            status=status,
            backend_used=backend_used,
            error=message,
            clarification_required=clarification_required,
            trace=trace or [],
        )


class BackendHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str
    status: HealthStatus
    loaded: bool
    detail: str = ""
    model_reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
