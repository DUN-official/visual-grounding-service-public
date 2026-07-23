"""Deterministic complexity-aware backend routing policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .schemas import BackendHealth, GroundingRequest, HealthStatus, ReasoningComplexity
from .task_parser import normalize_text, parse_grounding_prompt


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    yolo_target_aliases: dict[str, list[str]] = Field(default_factory=dict)
    remote_fallback_enabled: bool = True
    guided_fallback_to_owlvit: bool = True


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_backend: str | None
    fallback_order: list[str] = Field(default_factory=list)
    reason: str
    relation_detected: bool
    normalized_target: str
    reasoning_complexity: str = ReasoningComplexity.SIMPLE
    guidance_reasons: list[str] = Field(default_factory=list)


class GroundingRouter:
    def __init__(self, policy: RoutingPolicy):
        self.policy = policy

    def route(self, request: GroundingRequest, health: dict[str, BackendHealth]) -> RoutingDecision:
        parsed = parse_grounding_prompt(request.instruction)
        target = normalize_text(request.target_object or parsed.target_object)
        guided = bool(request.requires_guided_reasoning or parsed.requires_guided_reasoning)
        reasons = list(dict.fromkeys([
            *parsed.guidance_reasons,
            *(request.metadata.get("guidance_reasons", []) if isinstance(request.metadata, dict) else []),
        ]))
        relation = bool(request.relations or parsed.relations)
        complexity = str(
            request.reasoning_complexity
            if request.requires_guided_reasoning
            else parsed.reasoning_complexity
        )

        if request.preferred_backend:
            proposed = [request.preferred_backend, "remote", "gpt_guided_owlvit", "owlvit", "yolo"]
            order = self._healthy_order(proposed, health)
            return RoutingDecision(
                selected_backend=order[0] if order else None,
                fallback_order=order,
                reason=f"preferred backend requested: {request.preferred_backend}",
                relation_detected=relation,
                normalized_target=target,
                reasoning_complexity=complexity,
                guidance_reasons=reasons,
            )

        if guided:
            proposed = ["gpt_guided_owlvit", "remote"]
            if self.policy.guided_fallback_to_owlvit:
                proposed.append("owlvit")
            reason = "guided reasoning required: " + ", ".join(reasons or ["instruction complexity"])
        elif self._target_supported_by_yolo(target):
            proposed = ["yolo", "remote", "owlvit"]
            reason = "simple target is supported by the configured YOLO vocabulary"
        else:
            proposed = ["owlvit", "remote", "gpt_guided_owlvit"]
            reason = "simple open-vocabulary target requires OWL ViT"

        if not self.policy.remote_fallback_enabled:
            proposed = [name for name in proposed if name != "remote"]
        order = self._healthy_order(proposed, health)
        return RoutingDecision(
            selected_backend=order[0] if order else None,
            fallback_order=order,
            reason=reason,
            relation_detected=relation,
            normalized_target=target,
            reasoning_complexity=complexity,
            guidance_reasons=reasons,
        )

    def _target_supported_by_yolo(self, target: str) -> bool:
        if not target:
            return False
        for canonical, aliases in self.policy.yolo_target_aliases.items():
            terms = {normalize_text(canonical), *{normalize_text(alias) for alias in aliases}}
            if target in terms:
                return True
        return False

    @staticmethod
    def _ready(name: str, health: dict[str, BackendHealth]) -> bool:
        state = health.get(name)
        return bool(state and state.status == HealthStatus.READY and state.loaded)

    def _healthy_order(self, names: list[str], health: dict[str, BackendHealth]) -> list[str]:
        output = []
        for name in names:
            if name not in output and self._ready(name, health):
                output.append(name)
        return output
