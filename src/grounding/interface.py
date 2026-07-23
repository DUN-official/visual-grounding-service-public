from abc import ABC, abstractmethod
import threading
import time

from .exceptions import BackendNotReadyError
from .schemas import BackendHealth, GroundingResult, HealthStatus, TraceEvent

class GroundingBackend(ABC):
    def __init__(self, name):
        self.name = name
        self._started = False
        self._health_detail = "not started"
        self._model_reference = None
        self._inference_lock = threading.RLock()

    @abstractmethod
    def startup(self):
        """Load models or clients once."""

    def shutdown(self):
        self._started = False
        self._health_detail = "stopped"

    def health(self):
        return BackendHealth(
            backend=self.name,
            status=HealthStatus.READY if self._started else HealthStatus.UNAVAILABLE,
            loaded=self._started,
            detail=self._health_detail,
            model_reference=self._model_reference,
        )

    def ground(self, request):
        if not self._started:
            raise BackendNotReadyError(f"{self.name} has not completed startup")
        started = time.perf_counter()
        try:
            with self._inference_lock:
                result = self._ground_impl(request)
        except Exception as exc:
            result = GroundingResult.failure(
                request,
                backend_used=self.name,
                message=f"{type(exc).__name__}: {exc}",
                trace=[TraceEvent(stage=self.name, status="error", message=str(exc))],
            )
        result.backend_used = result.backend_used or self.name
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    @abstractmethod
    def _ground_impl(self, request):
        pass

    def supports(self, request):
        return True
