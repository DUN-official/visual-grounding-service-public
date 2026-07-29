"""Lazy, thread-safe access to the shared grounding models."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

from ..config import GroundingConfig, load_config, project_path
from ..openai_context import use_openai_api_key
from ..provisioning import (
    DEFAULT_OWLVIT_REPOSITORY,
    DEFAULT_OWLVIT_REVISION,
    DEFAULT_YOLO_MODEL,
    provision_owlvit,
    provision_yolo,
)
from ..schemas import GroundingRequest, GroundingResult, PerformanceMode
from ..services.local_service import LocalGroundingService
from ..task_parser import normalize_text, parse_grounding_prompt
from ..video.service_adapter import LocalGroundingServiceAdapter


class ServiceRuntime:
    """Own shared local models without sharing visitor credentials."""

    def __init__(
        self,
        service: LocalGroundingService,
        *,
        config: GroundingConfig,
        config_path: Path,
    ) -> None:
        self.service = service
        self.config = config
        self.config_path = config_path
        self.project_root = Path(config.service.project_root)
        self.lock = RLock()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        auto_provision: bool = False,
    ) -> "ServiceRuntime":
        # auto_provision remains accepted for compatibility. Models are now
        # provisioned on first use instead of during application startup.
        del auto_provision
        path = Path(config_path).expanduser().resolve()
        config = load_config(path)
        service = LocalGroundingService.from_config(config)
        service._started = True
        return cls(service, config=config, config_path=path)

    def ground(
        self,
        request: GroundingRequest,
        *,
        api_key: str | None = None,
    ) -> GroundingResult:
        backend_name = self.select_backend(
            request.preferred_backend,
            request.instruction,
            api_key=api_key,
            performance_mode=request.performance_mode,
        )
        self.ensure_backend(backend_name, api_key=api_key)
        with self.lock, use_openai_api_key(api_key):
            return self.service.ground(request)

    def select_backend(
        self,
        preferred_backend: str | None,
        instruction: str,
        *,
        api_key: str | None = None,
        performance_mode: PerformanceMode | str = PerformanceMode.BALANCED,
    ) -> str:
        if preferred_backend:
            return preferred_backend

        mode = (
            performance_mode
            if isinstance(performance_mode, PerformanceMode)
            else PerformanceMode(performance_mode)
        )
        if mode in {PerformanceMode.QUALITY, PerformanceMode.ACCURATE}:
            if not (api_key or os.environ.get("OPENAI_API_KEY")):
                raise ValueError("Quality mode requires an OpenAI API key.")
            return "gpt_guided_owlvit"

        parsed = parse_grounding_prompt(instruction)
        if parsed.requires_guided_reasoning and api_key:
            return "gpt_guided_owlvit"
        target = normalize_text(parsed.target_object)
        if self.service.router._target_supported_by_yolo(target):
            return "yolo"
        return "owlvit"

    def prepare_backend(
        self,
        preferred_backend: str | None,
        instruction: str,
        *,
        api_key: str | None = None,
        performance_mode: PerformanceMode | str = PerformanceMode.BALANCED,
    ) -> str:
        backend_name = self.select_backend(
            preferred_backend,
            instruction,
            api_key=api_key,
            performance_mode=performance_mode,
        )
        self.ensure_backend(backend_name, api_key=api_key)
        return backend_name
    def ensure_backend(
        self,
        backend_name: str,
        *,
        api_key: str | None = None,
    ) -> None:
        if backend_name not in self.service.backends:
            raise ValueError(f"backend is not configured: {backend_name}")
        if backend_name == "gpt_guided_owlvit" and not (
            api_key or os.environ.get("OPENAI_API_KEY")
        ):
            raise ValueError("Enter an OpenAI API key to use GPT-guided OWL-ViT.")

        with self.lock, use_openai_api_key(api_key):
            if backend_name == "gpt_guided_owlvit":
                self._ensure_started("owlvit")
                if (
                    self.config.backends.gpt_guided_owlvit.use_yolo_first
                    and "yolo" in self.service.backends
                ):
                    self._ensure_started("yolo")
            self._ensure_started(backend_name)

    def _ensure_started(self, backend_name: str) -> None:
        backend = self.service.backends[backend_name]
        if getattr(backend, "_started", False):
            return
        self._provision(backend_name)
        try:
            backend.startup()
            self.service.startup_errors.pop(backend_name, None)
        except Exception as exc:
            self.service.startup_errors[backend_name] = f"{type(exc).__name__}: {exc}"
            raise

    def _provision(self, backend_name: str) -> None:
        if backend_name == "yolo":
            settings = self.config.backends.yolo
            destination = project_path(self.config, settings.weights_path)
            if destination is not None and not destination.exists():
                provision_yolo(
                    destination,
                    model_name=os.environ.get(
                        "GROUNDING_YOLO_MODEL",
                        DEFAULT_YOLO_MODEL,
                    ),
                )
        elif backend_name == "owlvit":
            settings = self.config.backends.owlvit
            destination = project_path(self.config, settings.model_path)
            weight = destination / "model.safetensors" if destination else None
            if destination is not None and weight is not None and not weight.exists():
                provision_owlvit(
                    destination,
                    repository=os.environ.get(
                        "GROUNDING_OWLVIT_REPOSITORY",
                        DEFAULT_OWLVIT_REPOSITORY,
                    ),
                    revision=os.environ.get(
                        "GROUNDING_OWLVIT_REVISION",
                        DEFAULT_OWLVIT_REVISION,
                    ),
                )

    def health(self, *, api_key: str | None = None) -> dict[str, dict]:
        with self.lock, use_openai_api_key(api_key):
            return {
                name: state.model_dump(mode="json")
                for name, state in self.service.health().items()
            }

    def video_adapter(
        self,
        *,
        api_key: str | None = None,
    ) -> LocalGroundingServiceAdapter:
        return LocalGroundingServiceAdapter(
            self.service,
            lock=self.lock,
            api_key=api_key,
            prepare_backend=self.prepare_backend,
        )

    def shutdown(self) -> None:
        with self.lock:
            self.service.shutdown()
