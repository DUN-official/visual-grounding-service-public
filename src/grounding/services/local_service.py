"""Persistent in-process grounding service with stage-level timing."""

from __future__ import annotations

import time

from ..backends import GPTGuidedOWLViTBackend, OwlViTBackend, RemoteBackend, YoloBackend
from ..image_utils import load_pil_image
from ..llm_task_parser import parse_task_with_service
from ..segmentation import segment_from_box
from ..config import GroundingConfig, project_path
from ..evaluation.logger import GroundingJSONLLogger
from ..interface import GroundingBackend
from ..router import GroundingRouter
from ..schemas import (
    GroundingRequest,
    GroundingResult,
    GroundingStatus,
    QuantityIntent,
    ReasoningComplexity,
    SpatialConstraint,
    TraceEvent,
)
from ..task_parser import parse_grounding_prompt


class LocalGroundingService:
    def __init__(self, *, backends, router, logger=None, fail_startup_when_no_backend=True):
        self.backends: dict[str, GroundingBackend] = backends
        self.router = router
        self.logger = logger
        self.fail_startup_when_no_backend = fail_startup_when_no_backend
        self.startup_errors: dict[str, str] = {}
        self._started = False

    @classmethod
    def from_config(cls, config: GroundingConfig) -> "LocalGroundingService":
        allowed_roots = [project_path(config, value) for value in config.service.allowed_image_roots]
        allowed_roots = [path for path in allowed_roots if path is not None]
        max_bytes = config.service.max_image_bytes
        backends = {}

        yolo = None
        settings = config.backends.yolo
        if settings.enabled:
            yolo = YoloBackend(
                weights_path=project_path(config, settings.weights_path),
                class_aliases=settings.class_aliases,
                confidence_threshold=settings.confidence_threshold,
                image_size=settings.image_size,
                device=settings.device,
                max_candidates=settings.max_candidates,
                warmup_on_startup=settings.warmup_on_startup,
                allowed_image_roots=allowed_roots,
                max_image_bytes=max_bytes,
            )
            backends["yolo"] = yolo

        owl = None
        settings = config.backends.owlvit
        if settings.enabled:
            owl = OwlViTBackend(
                model_path=project_path(config, settings.model_path),
                device=settings.device,
                thresholds=settings.thresholds,
                top_k=settings.top_k,
                max_pairwise_iou=settings.max_pairwise_iou,
                max_image_width=settings.max_image_width,
                use_fp16=settings.use_fp16,
                warmup_on_startup=settings.warmup_on_startup,
                prompt_aliases=settings.prompt_aliases,
                allowed_image_roots=allowed_roots,
                max_image_bytes=max_bytes,
            )
            backends["owlvit"] = owl

        settings = config.backends.gpt_guided_owlvit
        if settings.enabled:
            if owl is None:
                raise ValueError("gpt_guided_owlvit requires owlvit.enabled=true")
            backends["gpt_guided_owlvit"] = GPTGuidedOWLViTBackend(
                owlvit_backend=owl,
                yolo_backend=yolo,
                openai_model=settings.openai_model,
                api_key_env=settings.api_key_env,
                allow_session_api_key=settings.allow_session_api_key,
                image_detail=settings.image_detail,
                top_k_initial=settings.top_k_initial,
                top_k_refined=settings.top_k_refined,
                local_crop_margin=settings.local_crop_margin,
                use_yolo_first=settings.use_yolo_first,
                enable_local_geometry=settings.enable_local_geometry,
                local_geometry_min_score=settings.local_geometry_min_score,
                local_geometry_min_margin=settings.local_geometry_min_margin,
                single_candidate_confidence=settings.single_candidate_confidence,
                skip_gpt_when_unambiguous=settings.skip_gpt_when_unambiguous,
                gpt_image_max_width=settings.gpt_image_max_width,
                gpt_jpeg_quality=settings.gpt_jpeg_quality,
                openai_timeout_seconds=settings.openai_timeout_seconds,
                openai_max_retries=settings.openai_max_retries,
                quality_thresholds=settings.quality_thresholds,
                quality_use_original_image=settings.quality_use_original_image,
                maximum_edge_adjustment=settings.maximum_edge_adjustment,
                maximum_edge_contraction=settings.maximum_edge_contraction,
                minimum_adjustment_confidence=settings.minimum_adjustment_confidence,
                minimum_adjusted_box_overlap=settings.minimum_adjusted_box_overlap,
                minimum_adjusted_area_ratio=settings.minimum_adjusted_area_ratio,
                maximum_adjusted_area_ratio=settings.maximum_adjusted_area_ratio,
                default_box_padding=settings.default_box_padding,
                debug_output_dir=project_path(config, settings.debug_output_dir),
                allowed_image_roots=allowed_roots,
                max_image_bytes=max_bytes,
            )

        settings = config.backends.remote
        if settings.enabled:
            backends["remote"] = RemoteBackend(
                endpoint=settings.endpoint,
                api_key_env=settings.api_key_env,
                allow_session_api_key=settings.allow_session_api_key,
                healthcheck=settings.healthcheck,
                allowed_image_roots=allowed_roots,
                max_image_bytes=max_bytes,
            )

        return cls(
            backends=backends,
            router=GroundingRouter(config.router),
            logger=GroundingJSONLLogger(project_path(config, config.service.request_log)),
            fail_startup_when_no_backend=config.service.fail_startup_when_no_backend,
        )

    def startup(self):
        self.startup_errors.clear()
        priority = ["yolo", "owlvit", "gpt_guided_owlvit", "remote"]
        ordered = priority + [name for name in self.backends if name not in priority]
        for name in ordered:
            backend = self.backends.get(name)
            if backend is None:
                continue
            try:
                backend.startup()
            except Exception as exc:
                self.startup_errors[name] = f"{type(exc).__name__}: {exc}"
        ready_count = sum(state.loaded for state in self.health().values())
        if ready_count == 0 and self.fail_startup_when_no_backend:
            raise RuntimeError("no grounding backend started successfully: " + str(self.startup_errors))
        self._started = True

    def shutdown(self):
        for backend in reversed(list(self.backends.values())):
            try:
                backend.shutdown()
            except Exception:
                pass
        self._started = False

    def health(self):
        output = {name: backend.health() for name, backend in self.backends.items()}
        for name, error in self.startup_errors.items():
            if name in output:
                output[name].detail = error
        return output

    def prepare_request(self, request: GroundingRequest):
        started = time.perf_counter()
        local = parse_grounding_prompt(request.instruction)
        source_metadata = dict(request.metadata)
        parser_mode = str(source_metadata.get("parser_mode") or "llm").lower()
        if source_metadata.get("input_mode") == "video_frame":
            parser_mode = "local"
        structured = parse_task_with_service(
            self,
            request.instruction,
            parser_mode=parser_mode,
        )

        structured_relations = [
            SpatialConstraint(
                relation=str(item.get("relation") or "").strip(),
                anchor=str(item.get("anchor_object") or "").strip(),
                raw_text=str(item.get("raw_text") or "").strip(),
            )
            for item in structured.relations
            if str(item.get("relation") or "").strip()
            and str(item.get("anchor_object") or "").strip()
        ]
        try:
            structured_quantity = QuantityIntent(structured.quantity)
        except ValueError:
            structured_quantity = local.quantity
        try:
            structured_complexity = ReasoningComplexity(structured.reasoning_complexity)
        except ValueError:
            structured_complexity = local.reasoning_complexity

        relations = structured_relations or list(local.relations)
        attributes = list(dict.fromkeys([*structured.attributes, *local.attributes]))
        anchors = list(dict.fromkeys([*structured.anchor_objects, *local.anchor_objects]))
        quantity = (
            structured_quantity
            if structured_quantity != QuantityIntent.UNKNOWN
            else local.quantity
        )
        minimum_count = max(local.minimum_count, structured.minimum_count)
        inferred_maximum = (
            10 if quantity in {QuantityIntent.MULTIPLE, QuantityIntent.ALL} else 1
        )
        guidance_reasons = list(local.guidance_reasons)
        if structured.negated_attributes:
            guidance_reasons.append("negation")
        if structured.parser_confidence < 0.65:
            guidance_reasons.append("low_parser_confidence")
        guidance_reasons = list(dict.fromkeys(guidance_reasons))

        updates = {}
        inferred = {
            "target_object": structured.target_object or local.target_object,
            "target_phrase": structured.target_phrase or local.target_phrase,
            "location_hint": " ".join(item.raw_text for item in relations).strip()
            or local.location_hint,
            "action": local.action,
            "quantity": quantity,
            "minimum_count": minimum_count,
            "maximum_results": inferred_maximum,
            "attributes": attributes,
            "relations": relations,
            "anchor_objects": anchors,
            "reasoning_complexity": structured_complexity,
            "requires_guided_reasoning": bool(
                structured.requires_guided_reasoning
                or local.requires_guided_reasoning
                or structured.negated_attributes
            ),
        }
        for key, value in inferred.items():
            current = getattr(request, key)
            if key in {"attributes", "relations", "anchor_objects"}:
                if not current and value:
                    updates[key] = value
            elif key in {"quantity", "reasoning_complexity"}:
                if current.value in {"unknown", "simple"} and value is not None:
                    updates[key] = value
            elif key in {"minimum_count", "maximum_results"}:
                if key not in request.model_fields_set and value != current:
                    updates[key] = value
            elif key == "requires_guided_reasoning":
                if not current and value:
                    updates[key] = value
            elif not current and value:
                updates[key] = value

        metadata = source_metadata
        metadata.update(
            {
                "parser_mode": parser_mode,
                "parser_source": structured.parser_source,
                "parser_confidence": structured.parser_confidence,
                "parser_fallback_reason": structured.fallback_reason,
                "candidate_prompts": list(structured.candidate_prompts),
                "negated_attributes": list(structured.negated_attributes),
                "anchor_phrases": list(structured.anchor_phrases),
                "recommended_backend": structured.recommended_backend,
                "guidance_reasons": guidance_reasons,
            }
        )
        updates["metadata"] = metadata
        effective = request.model_copy(update=updates)
        duration = (time.perf_counter() - started) * 1000.0
        trace = TraceEvent(
            stage="prompt_parser", status="parsed", duration_ms=duration,
            message="validated structured task and preserved grounding constraints",
            data={
                "target_object": effective.target_object,
                "target_phrase": effective.target_phrase,
                "location_hint": effective.location_hint,
                "action": effective.action,
                "quantity": effective.quantity,
                "minimum_count": effective.minimum_count,
                "maximum_results": effective.maximum_results,
                "attributes": effective.attributes,
                "relations": [item.model_dump(mode="json") for item in effective.relations],
                "anchor_objects": effective.anchor_objects,
                "reasoning_complexity": effective.reasoning_complexity,
                "requires_guided_reasoning": effective.requires_guided_reasoning,
                "guidance_reasons": guidance_reasons,
                "parser_mode": parser_mode,
                "parser_source": structured.parser_source,
                "parser_confidence": structured.parser_confidence,
                "parser_fallback_reason": structured.fallback_reason,
                "candidate_prompts": structured.candidate_prompts,
                "negated_attributes": structured.negated_attributes,
                "fields_inferred": sorted(updates),
            },
        )
        return effective, trace

    def route(self, request):
        effective, _ = self.prepare_request(request)
        return self.router.route(effective, self.health())

    def ground(self, request: GroundingRequest) -> GroundingResult:
        if not self._started:
            raise RuntimeError("service has not completed startup")
        started = time.perf_counter()
        effective_request, parser_trace = self.prepare_request(request)
        route_started = time.perf_counter()
        decision = self.router.route(effective_request, self.health())
        route_duration = (time.perf_counter() - route_started) * 1000.0
        route_trace = TraceEvent(
            stage="router",
            status="selected" if decision.selected_backend else "unavailable",
            duration_ms=route_duration,
            message=decision.reason,
            data=decision.model_dump(mode="json"),
        )
        if not decision.fallback_order:
            result = GroundingResult.failure(
                effective_request,
                status=GroundingStatus.BACKEND_UNAVAILABLE,
                message="no healthy backend can handle this request",
                clarification_required=True,
                trace=[parser_trace, route_trace],
            )
            self._finish(effective_request, result, started, parser_trace, route_trace)
            return result

        last_result = None
        attempted = []
        for backend_name in decision.fallback_order:
            elapsed = (time.perf_counter() - started) * 1000.0
            if elapsed >= effective_request.maximum_latency_ms:
                result = GroundingResult.failure(
                    effective_request,
                    status=GroundingStatus.TIMEOUT,
                    backend_used=backend_name,
                    message="deadline reached before backend execution",
                    clarification_required=True,
                    trace=[parser_trace, route_trace],
                )
                self._finish(effective_request, result, started, parser_trace, route_trace)
                return result
            attempted.append(backend_name)
            result = self.backends[backend_name].ground(effective_request)
            result.trace.insert(0, route_trace)
            result.trace.insert(0, parser_trace)
            result.latency_ms = (time.perf_counter() - started) * 1000.0
            result.metadata.setdefault("attempted_backends", list(attempted))
            if result.latency_ms > effective_request.maximum_latency_ms:
                timeout_result = GroundingResult.failure(
                    effective_request,
                    status=GroundingStatus.TIMEOUT,
                    backend_used=backend_name,
                    message="backend completed after request deadline",
                    clarification_required=True,
                    trace=result.trace,
                )
                timeout_result.metadata = result.metadata
                self._finish(effective_request, timeout_result, started, parser_trace, route_trace)
                return timeout_result
            if result.status == GroundingStatus.SUCCESS:
                self._maybe_attach_segmentation(effective_request, result)
                self._finish(effective_request, result, started, parser_trace, route_trace)
                return result
            last_result = result

        result = last_result or GroundingResult.failure(
            effective_request,
            status=GroundingStatus.BACKEND_UNAVAILABLE,
            message="all routed backends failed",
            clarification_required=True,
            trace=[parser_trace, route_trace],
        )
        self._finish(effective_request, result, started, parser_trace, route_trace)
        return result


    def _maybe_attach_segmentation(self, request: GroundingRequest, result: GroundingResult) -> None:
        metadata = dict(request.metadata)
        if not bool(metadata.get("return_segmentation")):
            return
        if result.status != GroundingStatus.SUCCESS or result.bbox_xyxy is None:
            return
        try:
            import numpy as np

            image = load_pil_image(request.image)
            image_rgb = np.array(image.convert("RGB"))
            prediction_items = list(result.predictions) if result.predictions else []
            if prediction_items:
                for prediction in prediction_items:
                    segmentation = segment_from_box(image_rgb, prediction.bbox_xyxy)
                    prediction.metadata["segmentation"] = segmentation.payload
                result.metadata["segmentation_enabled"] = True
                result.metadata["segmentation_method"] = "grabcut_box_prompt"
            else:
                segmentation = segment_from_box(image_rgb, result.bbox_xyxy)
                result.metadata["segmentation_enabled"] = True
                result.metadata["segmentation_method"] = "grabcut_box_prompt"
                result.metadata["segmentation"] = segmentation.payload
        except Exception as exc:
            result.metadata["segmentation_enabled"] = False
            result.metadata["segmentation_error"] = f"{type(exc).__name__}: {exc}"

    def _finish(self, request, result, started, parser_trace, route_trace):
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        stage = result.metadata.setdefault("stage_latencies_ms", {})
        stage.setdefault("parser", parser_trace.duration_ms or 0.0)
        stage.setdefault("router", route_trace.duration_ms or 0.0)
        stage["total"] = result.latency_ms
        self._log(request, result)

    def _log(self, request, result):
        if self.logger is not None:
            self.logger.log(request, result)
