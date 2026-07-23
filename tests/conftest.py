from __future__ import annotations

from grounding.interface import GroundingBackend
from grounding.schemas import BBoxXYXY, GroundingResult, GroundingStatus, TraceEvent


class FakeGroundingBackend(GroundingBackend):
    def __init__(self, name="owlvit"):
        super().__init__(name)

    def startup(self):
        self._started = True
        self._health_detail = "fake backend ready"

    def _ground_impl(self, request):
        return GroundingResult(
            request_id=request.request_id,
            status=GroundingStatus.SUCCESS,
            bbox_xyxy=BBoxXYXY(x_min=10, y_min=20, x_max=70, y_max=80),
            confidence=0.91,
            relation_match=True,
            backend_used=self.name,
            trace=[TraceEvent(stage="fake_inference", message="test result")],
        )
