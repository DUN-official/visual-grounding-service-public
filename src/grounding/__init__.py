"""Runtime visual grounding service."""

from .interface import GroundingBackend
from .router import GroundingRouter, RoutingDecision
from .schemas import (
    BackendHealth,
    BBoxXYXY,
    GroundingCandidate,
    GroundingRequest,
    GroundingResult,
    GroundingStatus,
    HealthStatus,
    ImagePayload,
    TraceEvent,
)

__version__ = "0.2.0"

__all__ = [
    "BackendHealth",
    "BBoxXYXY",
    "GroundingBackend",
    "GroundingCandidate",
    "GroundingRequest",
    "GroundingResult",
    "GroundingRouter",
    "GroundingStatus",
    "HealthStatus",
    "ImagePayload",
    "RoutingDecision",
    "TraceEvent",
]
