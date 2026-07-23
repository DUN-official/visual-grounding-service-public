"""Configuration schemas and path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .router import RoutingPolicy


class BackendBaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class YoloSettings(BackendBaseSettings):
    weights_path: str = "models/yolo/detector.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640
    device: str = "auto"
    max_candidates: int = 20
    warmup_on_startup: bool = True
    class_aliases: dict[str, list[str]] = Field(default_factory=dict)


class OwlViTSettings(BackendBaseSettings):
    model_path: str = "models/owlvit/owlvit-base-patch32"
    device: str = "auto"
    thresholds: list[float] = Field(default_factory=lambda: [0.05, 0.02, 0.01, 0.005])
    top_k: int = 6
    max_pairwise_iou: float = 0.85
    max_image_width: int = 960
    use_fp16: bool = True
    warmup_on_startup: bool = True


class GPTGuidedSettings(BackendBaseSettings):
    openai_model: str = "gpt-5"
    api_key_env: str = "OPENAI_API_KEY"
    allow_session_api_key: bool = False
    image_detail: str = "low"
    top_k_initial: int = 6
    top_k_refined: int = 4
    local_crop_margin: float = 0.20
    use_yolo_first: bool = True
    enable_local_geometry: bool = True
    local_geometry_min_score: float = 0.72
    local_geometry_min_margin: float = 0.12
    single_candidate_confidence: float = 0.72
    skip_gpt_when_unambiguous: bool = True
    gpt_image_max_width: int = 1024
    gpt_jpeg_quality: int = 78
    openai_timeout_seconds: float = 20.0
    openai_max_retries: int = 1
    maximum_edge_adjustment: float = 0.10
    maximum_edge_contraction: float = 0.06
    minimum_adjustment_confidence: float = 0.65
    minimum_adjusted_box_overlap: float = 0.55
    minimum_adjusted_area_ratio: float = 0.80
    maximum_adjusted_area_ratio: float = 1.60
    default_box_padding: float = 0.04
    debug_output_dir: str | None = None


class RemoteSettings(BackendBaseSettings):
    endpoint: str = "http://127.0.0.1:8000"
    api_key_env: str | None = None
    healthcheck: bool = True


class BackendSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    yolo: YoloSettings = Field(default_factory=YoloSettings)
    owlvit: OwlViTSettings = Field(default_factory=OwlViTSettings)
    gpt_guided_owlvit: GPTGuidedSettings = Field(default_factory=GPTGuidedSettings)
    remote: RemoteSettings = Field(default_factory=RemoteSettings)


class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_root: str = ".."
    request_log: str = "runtime/logs/grounding_requests.jsonl"
    allowed_image_roots: list[str] = Field(default_factory=list)
    max_image_bytes: int = 30 * 1024 * 1024
    fail_startup_when_no_backend: bool = True


class GroundingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    router: RoutingPolicy = Field(default_factory=RoutingPolicy)
    backends: BackendSettings = Field(default_factory=BackendSettings)


def load_config(path: str | Path) -> GroundingConfig:
    config_path = Path(path).expanduser().resolve()
    data = _expand_environment(json.loads(config_path.read_text(encoding="utf-8")))
    config = GroundingConfig.model_validate(data)
    root = Path(config.service.project_root).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    config.service.project_root = str(root)
    return config


def project_path(config: GroundingConfig, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config.service.project_root) / path
    return path.resolve()


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
