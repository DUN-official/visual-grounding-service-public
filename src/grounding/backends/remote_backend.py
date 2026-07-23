import base64
import os

from ..exceptions import RemoteBackendError
from ..image_utils import payload_bytes
from ..interface import GroundingBackend
from ..schemas import GroundingResult, ImagePayload


class RemoteBackend(GroundingBackend):
    def __init__(
        self, *, endpoint, api_key_env=None, healthcheck=True,
        allowed_image_roots=None, max_image_bytes=30 * 1024 * 1024,
    ):
        super().__init__("remote")
        self.endpoint = endpoint.rstrip("/")
        self.api_key_env = api_key_env
        self.healthcheck_enabled = healthcheck
        self.allowed_image_roots = allowed_image_roots
        self.max_image_bytes = max_image_bytes
        self._client = None
        self._headers = {}

    def startup(self):
        import httpx
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise EnvironmentError(f"missing remote API key: {self.api_key_env}")
            self._headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=self.endpoint, headers=self._headers, timeout=30.0
        )
        if self.healthcheck_enabled:
            response = self._client.get("/health")
            response.raise_for_status()
        self._started = True
        self._health_detail = "remote service reachable"
        self._model_reference = self.endpoint

    def shutdown(self):
        if self._client is not None:
            self._client.close()
        self._client = None
        super().shutdown()

    def _ground_impl(self, request):
        outbound = request.model_copy(deep=True)
        if outbound.image.path:
            raw, media_type = payload_bytes(
                outbound.image,
                allowed_roots=self.allowed_image_roots,
                max_bytes=self.max_image_bytes,
            )
            outbound.image = ImagePayload(
                base64_data=base64.b64encode(raw).decode("ascii"),
                media_type=media_type,
            )
        response = self._client.post(
            "/v1/ground",
            json=outbound.model_dump(mode="json"),
            timeout=max(1.0, request.maximum_latency_ms / 1000.0),
        )
        if response.status_code >= 400:
            raise RemoteBackendError(
                f"remote returned {response.status_code}: {response.text[:300]}"
            )
        return GroundingResult.model_validate(response.json())
