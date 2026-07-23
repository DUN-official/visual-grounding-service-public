"""One-shot LLM task parsing for video sessions.

The parser is intentionally separate from frame inference. It runs once when a
video session starts, preserves the user's original instruction unchanged, and
returns a small validated plan used only for routing and target/anchor safety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any

from .openai_context import resolve_openai_api_key
from .task_parser import normalize_text, parse_grounding_prompt


_BACKENDS = {"yolo", "owlvit", "gpt_guided_owlvit"}
_TRACK_ACTIONS = ("track", "follow", "monitor")


@dataclass(slots=True)
class VideoTaskPlan:
    instruction: str
    normalized_instruction: str
    action: str
    target_object: str | None
    target_phrase: str | None
    location_hint: str | None
    attributes: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    anchor_objects: list[str] = field(default_factory=list)
    anchor_phrases: list[str] = field(default_factory=list)
    requires_guided_reasoning: bool = False
    recommended_backend: str | None = None
    parser_source: str = "local"
    parser_confidence: float = 0.0
    parser_error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class LLMVideoTaskParser:
    """Parse one instruction once, with deterministic fallback.

    The OpenAI call is text-only and does not run per frame. The original
    instruction is never rewritten before being sent to the image-grounding
    pipeline.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("GROUNDING_TASK_PARSER_MODEL", "gpt-5")

    def parse(self, instruction: str, *, use_llm: bool = True) -> VideoTaskPlan:
        original = instruction.strip()
        if not original:
            raise ValueError("video instruction must not be empty")

        local_plan = self._local_plan(original)
        api_key = resolve_openai_api_key()
        if not use_llm or not api_key:
            if use_llm and not api_key:
                local_plan.parser_error = "OPENAI_API_KEY is unavailable; local parser used"
            return local_plan

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.model,
                input=self._prompt(original),
            )
            payload = self._extract_json(response.output_text)
            return self._validated_llm_plan(original, payload, local_plan)
        except Exception as exc:  # Fall back rather than blocking video processing.
            local_plan.parser_source = "local_fallback"
            local_plan.parser_error = f"{type(exc).__name__}: {exc}"
            return local_plan

    @staticmethod
    def _normalize_tracking_action(instruction: str) -> str:
        normalized = instruction.strip()
        lowered = normalized.lower()
        for verb in _TRACK_ACTIONS:
            if lowered == verb:
                return "find"
            prefix = verb + " "
            if lowered.startswith(prefix):
                return "find " + normalized[len(prefix):]
        return normalized

    def _local_plan(self, instruction: str) -> VideoTaskPlan:
        normalized_instruction = self._normalize_tracking_action(instruction)
        parsed = parse_grounding_prompt(normalized_instruction)
        relations = [
            {
                "relation": item.relation,
                "anchor_object": item.anchor,
                "anchor_phrase": item.anchor,
                "anchor_attributes": [],
                "raw_text": item.raw_text,
            }
            for item in parsed.relations
        ]
        recommended = (
            "gpt_guided_owlvit" if parsed.requires_guided_reasoning else None
        )
        return VideoTaskPlan(
            instruction=instruction,
            normalized_instruction=normalized_instruction,
            action="track",
            target_object=parsed.target_object,
            target_phrase=parsed.target_phrase,
            location_hint=parsed.location_hint,
            attributes=list(parsed.attributes),
            relations=relations,
            anchor_objects=list(parsed.anchor_objects),
            anchor_phrases=list(parsed.anchor_objects),
            requires_guided_reasoning=parsed.requires_guided_reasoning,
            recommended_backend=recommended,
            parser_source="local",
            parser_confidence=0.70 if parsed.target_object else 0.35,
        )

    @staticmethod
    def _prompt(instruction: str) -> str:
        schema = {
            "target_object": "the object the user wants tracked, never the reference object",
            "target_phrase": "full target noun phrase",
            "target_attributes": ["attributes belonging only to the target"],
            "relations": [
                {
                    "relation": "canonical spatial relation",
                    "anchor_object": "reference object noun",
                    "anchor_phrase": "full reference phrase",
                    "anchor_attributes": ["attributes belonging only to anchor"],
                }
            ],
            "requires_guided_reasoning": True,
            "recommended_backend": "yolo | owlvit | gpt_guided_owlvit",
            "confidence": 0.0,
        }
        return (
            "Parse this visual tracking instruction. Distinguish the requested TARGET from "
            "all REFERENCE/ANCHOR objects. The object before a spatial relation is normally "
            "the target; the object after the relation is normally the anchor. Never swap "
            "them. Return JSON only, with exactly this structure:\n"
            + json.dumps(schema)
            + "\nInstruction: "
            + instruction
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        value = (text or "").strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value)
            value = re.sub(r"\s*```$", "", value)
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM parser did not return a JSON object")
        parsed = json.loads(value[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM parser response must be a JSON object")
        return parsed

    def _validated_llm_plan(
        self,
        instruction: str,
        payload: dict[str, Any],
        fallback: VideoTaskPlan,
    ) -> VideoTaskPlan:
        target_object = self._clean_text(payload.get("target_object")) or fallback.target_object
        target_phrase = self._clean_text(payload.get("target_phrase")) or target_object or fallback.target_phrase
        attributes = self._string_list(payload.get("target_attributes")) or list(fallback.attributes)

        relations: list[dict[str, Any]] = []
        anchors: list[str] = []
        anchor_phrases: list[str] = []
        for raw in payload.get("relations") or []:
            if not isinstance(raw, dict):
                continue
            anchor_object = self._clean_text(raw.get("anchor_object"))
            anchor_phrase = self._clean_text(raw.get("anchor_phrase")) or anchor_object
            relation = self._clean_text(raw.get("relation"))
            if not relation or not anchor_object:
                continue
            if target_object and normalize_text(anchor_object) == normalize_text(target_object):
                continue
            item = {
                "relation": relation,
                "anchor_object": anchor_object,
                "anchor_phrase": anchor_phrase,
                "anchor_attributes": self._string_list(raw.get("anchor_attributes")),
                "raw_text": f"{relation} {anchor_phrase or anchor_object}",
            }
            relations.append(item)
            anchors.append(anchor_object)
            if anchor_phrase:
                anchor_phrases.append(anchor_phrase)

        if not relations:
            relations = list(fallback.relations)
            anchors = list(fallback.anchor_objects)
            anchor_phrases = list(fallback.anchor_phrases)

        guided = bool(
            payload.get("requires_guided_reasoning")
            or relations
            or attributes
            or fallback.requires_guided_reasoning
        )
        backend = self._clean_text(payload.get("recommended_backend"))
        if backend not in _BACKENDS:
            backend = "gpt_guided_owlvit" if guided else fallback.recommended_backend
        if guided:
            backend = "gpt_guided_owlvit"

        confidence = payload.get("confidence", 0.85)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.85

        location_hint = " ".join(
            str(item.get("raw_text") or "").strip() for item in relations
        ).strip() or fallback.location_hint

        return VideoTaskPlan(
            instruction=instruction,
            normalized_instruction=self._normalize_tracking_action(instruction),
            action="track",
            target_object=target_object,
            target_phrase=target_phrase,
            location_hint=location_hint,
            attributes=attributes,
            relations=relations,
            anchor_objects=list(dict.fromkeys(anchors)),
            anchor_phrases=list(dict.fromkeys(anchor_phrases)),
            requires_guided_reasoning=guided,
            recommended_backend=backend,
            parser_source="llm",
            parser_confidence=confidence_value,
        )

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = normalize_text(value).strip(" .,:;!?\"")
        return cleaned or None

    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(item for item in (cls._clean_text(v) for v in value) if item))

@dataclass(slots=True)
class ParsedVideoTask:
    instruction: str
    target_object: str | None
    target_phrase: str | None
    attributes: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    anchor_objects: list[str] = field(default_factory=list)
    anchor_phrases: list[str] = field(default_factory=list)
    requires_guided_reasoning: bool = False
    recommended_backend: str | None = None
    parser_source: str = "local"
    parser_confidence: float = 0.0
    fallback_reason: str | None = None


def parse_task_with_service(
    service,
    instruction: str,
    *,
    parser_mode: str = "llm",
) -> ParsedVideoTask:
    """Parse one video instruction using the loaded GPT client when available."""

    parser = LLMVideoTaskParser()
    local = parser._local_plan(instruction)
    if parser_mode.lower() != "llm":
        return _from_local(local)

    backend = getattr(service, "backends", {}).get("gpt_guided_owlvit")
    client = None
    if backend is not None:
        client_factory = getattr(backend, "_client_for_request", None)
        if callable(client_factory):
            try:
                client = client_factory()
            except Exception:
                client = None
        else:
            client = getattr(backend, "_client", None)
    loaded = False
    if backend is not None:
        try:
            loaded = bool(backend.health().loaded)
        except Exception:
            loaded = bool(getattr(backend, "_loaded", False))
    if not loaded or client is None:
        parsed = _from_local(local)
        parsed.parser_source = "local_fallback"
        parsed.fallback_reason = "GPT task parser is unavailable; local parser used"
        return parsed

    try:
        response = client.responses.create(
            model=getattr(backend, "openai_model", parser.model),
            input=parser._prompt(instruction),
        )
        payload = parser._extract_json(response.output_text)
        return _from_service_payload(instruction, payload, local)
    except Exception as exc:
        parsed = _from_local(local)
        parsed.parser_source = "local_fallback"
        parsed.fallback_reason = f"{type(exc).__name__}: {exc}"
        return parsed


def _from_service_payload(
    instruction: str,
    payload: dict[str, Any],
    local: VideoTaskPlan,
) -> ParsedVideoTask:
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        target_payload = {}
    target_object = _clean(
        target_payload.get("object") or payload.get("target_object")
    ) or local.target_object
    target_phrase = _clean(
        target_payload.get("phrase") or payload.get("target_phrase")
    ) or target_object or local.target_phrase
    attributes = _strings(
        target_payload.get("attributes") or payload.get("target_attributes")
    ) or list(local.attributes)

    relations: list[dict[str, Any]] = []
    anchors: list[str] = []
    anchor_phrases: list[str] = []
    for raw in payload.get("relations") or []:
        if not isinstance(raw, dict):
            continue
        relation = _clean(raw.get("type") or raw.get("relation"))
        anchor_object = _clean(raw.get("anchor_object"))
        anchor_phrase = _clean(raw.get("anchor_phrase")) or anchor_object
        if not relation or not anchor_object:
            continue
        relations.append(
            {
                "relation": relation,
                "anchor_object": anchor_object,
                "anchor_phrase": anchor_phrase,
                "anchor_attributes": _strings(raw.get("anchor_attributes")),
                "raw_text": _clean(raw.get("raw_text")) or f"{relation} {anchor_phrase}",
            }
        )
        anchors.append(anchor_object)
        if anchor_phrase:
            anchor_phrases.append(anchor_phrase)

    if not relations:
        relations = list(local.relations)
        anchors = list(local.anchor_objects)
        anchor_phrases = list(local.anchor_phrases)

    fallback_reason = None
    normalized_target = normalize_text(target_object)
    normalized_anchors = {normalize_text(value) for value in anchors}
    if normalized_target and normalized_target in normalized_anchors:
        target_object = local.target_object
        target_phrase = local.target_phrase
        attributes = list(local.attributes)
        fallback_reason = (
            "Parsed target matched a reference anchor; local target retained"
        )

    guided = bool(
        payload.get("requires_instance_selection")
        or payload.get("requires_guided_reasoning")
        or relations
        or attributes
        or local.requires_guided_reasoning
    )
    backend = _clean(payload.get("recommended_backend"))
    if backend not in {"yolo", "owlvit", "gpt_guided_owlvit"}:
        backend = "gpt_guided_owlvit" if guided else local.recommended_backend
    if guided:
        backend = "gpt_guided_owlvit"

    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.85))))
    except (TypeError, ValueError):
        confidence = 0.85

    return ParsedVideoTask(
        instruction=instruction,
        target_object=target_object,
        target_phrase=target_phrase,
        attributes=attributes,
        relations=relations,
        anchor_objects=list(dict.fromkeys(anchors)),
        anchor_phrases=list(dict.fromkeys(anchor_phrases)),
        requires_guided_reasoning=guided,
        recommended_backend=backend,
        parser_source="llm" if fallback_reason is None else "local_fallback",
        parser_confidence=confidence,
        fallback_reason=fallback_reason,
    )


def _from_local(plan: VideoTaskPlan) -> ParsedVideoTask:
    return ParsedVideoTask(
        instruction=plan.instruction,
        target_object=plan.target_object,
        target_phrase=plan.target_phrase,
        attributes=list(plan.attributes),
        relations=list(plan.relations),
        anchor_objects=list(plan.anchor_objects),
        anchor_phrases=list(plan.anchor_phrases),
        requires_guided_reasoning=plan.requires_guided_reasoning,
        recommended_backend=plan.recommended_backend,
        parser_source=plan.parser_source,
        parser_confidence=plan.parser_confidence,
    )


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = normalize_text(value).strip(" .,:;!?\"")
    return cleaned or None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in (_clean(entry) for entry in value) if item))


__all__ = [
    "LLMVideoTaskParser",
    "ParsedVideoTask",
    "VideoTaskPlan",
    "parse_task_with_service",
]
