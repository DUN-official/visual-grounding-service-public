from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from PIL import ImageDraw

from ..evaluation.metrics import box_iou
from ..image_utils import image_to_data_url, load_pil_image
from ..interface import GroundingBackend
from ..openai_context import resolve_openai_api_key
from ..schemas import (
    BBoxXYXY, GroundingCandidate, GroundingPrediction, GroundingResult,
    GroundingStatus, HealthStatus, PerformanceMode, QuantityIntent, TraceEvent,
)
from ..spatial import clear_winner, rank_candidates_by_constraints
from ..task_parser import normalize_text


class GPTGuidedOWLViTBackend(GroundingBackend):
    """Mode-aware guided grounding with bounded visual refinement.

    Fast path:
        one YOLO scene pass -> local relation scoring -> return
    Open-vocabulary path:
        one OWL ViT pass -> local relation scoring -> return when decisive
    Balanced path:
        candidate generation -> one combined GPT selection/edge decision
    Quality path:
        OWL ViT candidates -> GPT selection or coarse region -> local OWL ViT
        refinement -> GPT refined selection -> bounded GPT edge review
    """

    def __init__(
        self, *, owlvit_backend, yolo_backend=None, openai_model,
        api_key_env="OPENAI_API_KEY", allow_session_api_key=False,
        image_detail="high",
        top_k_initial=5, top_k_refined=5, local_crop_margin=0.25,
        use_yolo_first=True, enable_local_geometry=True,
        local_geometry_min_score=0.72, local_geometry_min_margin=0.12,
        single_candidate_confidence=0.72, skip_gpt_when_unambiguous=True,
        gpt_image_max_width=1600, gpt_jpeg_quality=92,
        openai_timeout_seconds=60.0, openai_max_retries=1,
        quality_thresholds=None, quality_use_original_image=True,
        maximum_edge_adjustment=0.10, maximum_edge_contraction=0.06,
        minimum_adjustment_confidence=0.65,
        minimum_adjusted_box_overlap=0.55,
        minimum_adjusted_area_ratio=0.80,
        maximum_adjusted_area_ratio=1.60,
        default_box_padding=0.04,
        debug_output_dir=None, allowed_image_roots=None,
        max_image_bytes=30 * 1024 * 1024,
    ):
        super().__init__("gpt_guided_owlvit")
        self.owlvit_backend = owlvit_backend
        self.yolo_backend = yolo_backend
        self.openai_model = openai_model
        self.api_key_env = api_key_env
        self.allow_session_api_key = bool(allow_session_api_key)
        self.image_detail = image_detail
        self.top_k_initial = top_k_initial
        self.top_k_refined = top_k_refined
        self.local_crop_margin = local_crop_margin
        self.use_yolo_first = use_yolo_first
        self.enable_local_geometry = enable_local_geometry
        self.local_geometry_min_score = local_geometry_min_score
        self.local_geometry_min_margin = local_geometry_min_margin
        self.single_candidate_confidence = single_candidate_confidence
        self.skip_gpt_when_unambiguous = skip_gpt_when_unambiguous
        self.gpt_image_max_width = gpt_image_max_width
        self.gpt_jpeg_quality = gpt_jpeg_quality
        self.openai_timeout_seconds = openai_timeout_seconds
        self.openai_max_retries = openai_max_retries
        self.quality_thresholds = quality_thresholds or [0.05, 0.02, 0.01, 0.005, 0.0]
        self.quality_use_original_image = bool(quality_use_original_image)
        self.maximum_edge_adjustment = maximum_edge_adjustment
        self.maximum_edge_contraction = maximum_edge_contraction
        self.minimum_adjustment_confidence = minimum_adjustment_confidence
        self.minimum_adjusted_box_overlap = minimum_adjusted_box_overlap
        self.minimum_adjusted_area_ratio = minimum_adjusted_area_ratio
        self.maximum_adjusted_area_ratio = maximum_adjusted_area_ratio
        self.default_box_padding = default_box_padding
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        self.allowed_image_roots = allowed_image_roots
        self.max_image_bytes = max_image_bytes
        self._client = None

    def startup(self):
        if not self.owlvit_backend.health().loaded:
            raise RuntimeError("GPT-guided backend requires a started OWL ViT backend")
        api_key = resolve_openai_api_key(self.api_key_env)
        if not api_key and not self.allow_session_api_key:
            raise EnvironmentError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        if self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        yolo_state = (
            "ready"
            if self.yolo_backend and self.yolo_backend.health().loaded
            else "unavailable"
        )
        self._started = True
        credential_mode = "server key" if api_key else "session key required"
        self._health_detail = (
            "fast, balanced, and quality grounding paths ready; "
            f"YOLO fast path {yolo_state}; {credential_mode}"
        )
        self._model_reference = self.openai_model

    def shutdown(self):
        self._client = None
        super().shutdown()

    def health(self):
        state = super().health()
        if (
            self._started
            and self.allow_session_api_key
            and not resolve_openai_api_key(self.api_key_env)
        ):
            state.status = HealthStatus.UNAVAILABLE
            state.loaded = False
            state.detail = "OpenAI API key required for this session"
        return state

    def _client_for_request(self):
        if self._client is not None:
            return self._client
        api_key = resolve_openai_api_key(self.api_key_env)
        if not api_key:
            raise EnvironmentError("OpenAI API key is required for GPT-guided grounding")
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            timeout=self.openai_timeout_seconds,
            max_retries=self.openai_max_retries,
        )
    def _ground_impl(self, request):
        trace = []
        timings = {}
        pipeline_path = []
        gpt_request_count = 0

        started = time.perf_counter()
        image = load_pil_image(
            request.image,
            allowed_roots=self.allowed_image_roots,
            max_bytes=self.max_image_bytes,
        )
        timings["image_decode"] = self._elapsed(started)
        trace.append(TraceEvent(
            stage="image_decode", duration_ms=timings["image_decode"],
            data={"image_size": [image.width, image.height]},
        ))

        if request.performance_mode in {PerformanceMode.QUALITY, PerformanceMode.ACCURATE}:
            return self._quality_ground(
                request=request,
                image=image,
                trace=trace,
                timings=timings,
            )

        candidates = []
        anchors_by_name = {}
        candidate_source = None

        # A single YOLO scene pass can provide both target and anchor objects.
        if self._can_use_yolo(request):
            yolo_started = time.perf_counter()
            scene = self.yolo_backend.detect_scene(image)
            timings["yolo_scene"] = self._elapsed(yolo_started)
            candidates = self.yolo_backend.filter_candidates(
                scene, request.target_object, top_k=self.top_k_initial
            )
            for anchor_name in request.anchor_objects:
                anchors_by_name[normalize_text(anchor_name)] = self.yolo_backend.filter_candidates(
                    scene, anchor_name, top_k=self.top_k_initial
                )
            candidate_source = "yolo"
            pipeline_path.append("yolo_scene")
            trace.append(TraceEvent(
                stage="yolo_scene_candidates",
                duration_ms=timings["yolo_scene"],
                message=f"generated {len(candidates)} target candidates in one scene pass",
                data={
                    "target_candidate_count": len(candidates),
                    "anchor_candidate_counts": {
                        key: len(value) for key, value in anchors_by_name.items()
                    },
                },
            ))

        if not candidates:
            owl_started = time.perf_counter()
            candidates = self.owlvit_backend.detect_candidates(
                request, image=image, top_k=self.top_k_initial
            )
            timings["owlvit_initial"] = self._elapsed(owl_started)
            candidate_source = "owlvit"
            pipeline_path.append("owlvit_initial")
            trace.append(TraceEvent(
                stage="owlvit_initial_candidates",
                duration_ms=timings["owlvit_initial"],
                message=f"generated {len(candidates)} open-vocabulary candidates",
                data={"candidate_count": len(candidates)},
            ))

        if not candidates:
            return GroundingResult.failure(
                request, backend_used=self.name,
                message="no local detector produced a target candidate",
                clarification_required=True, trace=trace,
            )

        ranked = []
        if request.relations and self.enable_local_geometry:
            geometry_started = time.perf_counter()
            ranked = rank_candidates_by_constraints(
                candidates, request.relations, anchors_by_name,
                (image.width, image.height),
            )
            timings["relation_scoring"] = self._elapsed(geometry_started)
            trace.append(TraceEvent(
                stage="local_relation_scoring",
                duration_ms=timings["relation_scoring"],
                message="scored candidates against parsed spatial constraints",
                data={
                    "scores": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "combined_score": round(combined, 4),
                            "relation_score": round(relation_score, 4),
                        }
                        for combined, relation_score, candidate in ranked
                    ]
                },
            ))

        local_selection = self._local_fast_selection(request, candidates, ranked)
        if local_selection:
            pipeline_path.append("local_fast_return")
            return self._success_result(
                request=request,
                image=image,
                selected=local_selection,
                candidates=candidates,
                trace=trace,
                timings=timings,
                pipeline_path=pipeline_path,
                gpt_request_count=0,
                decision_reason="local confidence/geometry gate resolved the request",
                relation_match=True if request.relations else None,
            )

        if request.performance_mode == PerformanceMode.FAST:
            best = ranked[0][2] if ranked else candidates[0]
            return GroundingResult.failure(
                request,
                status=GroundingStatus.CLARIFICATION_REQUIRED,
                backend_used=self.name,
                message=(
                    "fast mode could not resolve ambiguity without GPT; "
                    f"best local candidate was {best.candidate_id}"
                ),
                clarification_required=True,
                trace=trace,
            )

        overlay = self._candidate_overlay(image, candidates, anchors_by_name)
        gpt_started = time.perf_counter()
        decision = self._gpt_final_decision(request, overlay, candidates)
        timings["gpt_final_decision"] = self._elapsed(gpt_started)
        gpt_request_count = 1
        pipeline_path.append("gpt_final_decision")
        trace.append(TraceEvent(
            stage="gpt_final_decision",
            duration_ms=timings["gpt_final_decision"],
            message=str(decision.get("reason", "")),
            data={**decision, "gpt_request_count": 1},
        ))

        selected_with_decisions = self._selected_candidates_from_decision(
            candidates, decision, request.maximum_results
        )
        if not selected_with_decisions:
            return GroundingResult.failure(
                request,
                status=GroundingStatus.CLARIFICATION_REQUIRED,
                backend_used=self.name,
                message="GPT could not select a valid candidate",
                clarification_required=True,
                trace=trace,
            )

        selected = []
        adjustment_gates = []
        for candidate, item_decision in selected_with_decisions:
            final_box, applied, gate = self._apply_safe_adjustment(
                candidate.bbox_xyxy, item_decision, image.width, image.height
            )
            selected.append((candidate, final_box, item_decision))
            adjustment_gates.append({
                "candidate_id": candidate.candidate_id,
                "adjustment_applied": applied,
                "gate": gate,
            })

        # Accurate mode adds one local OWL crop pass after GPT selection, but no
        # additional GPT call. Balanced mode returns immediately after one GPT call.
        if request.performance_mode == PerformanceMode.ACCURATE and len(selected) == 1:
            refinement_started = time.perf_counter()
            selected = [self._local_refine_after_selection(request, image, selected[0])]
            timings["owlvit_local_refinement"] = self._elapsed(refinement_started)
            pipeline_path.append("owlvit_local_refinement")
            trace.append(TraceEvent(
                stage="owlvit_local_refinement",
                duration_ms=timings["owlvit_local_refinement"],
                message="refined selected region without another GPT request",
            ))

        predictions = []
        for candidate, final_box, item_decision in selected:
            confidence = float(item_decision.get("confidence", candidate.confidence) or candidate.confidence)
            predictions.append(GroundingPrediction(
                bbox_xyxy=final_box,
                confidence=max(0.0, min(1.0, confidence)),
                label=candidate.label,
                relation_match=item_decision.get("relation_match"),
                candidate_id=candidate.candidate_id,
                metadata={"source": candidate.source},
            ))

        final_overlay = self._prediction_overlay(image, predictions)
        self._save_debug(request, overlay, final_overlay)
        relation_match = all(p.relation_match is not False for p in predictions)
        return GroundingResult(
            request_id=request.request_id,
            status=GroundingStatus.SUCCESS,
            bbox_xyxy=predictions[0].bbox_xyxy,
            predictions=predictions,
            confidence=predictions[0].confidence,
            relation_match=relation_match,
            backend_used=self.name,
            candidates=candidates,
            trace=trace,
            metadata={
                "pipeline_path": pipeline_path,
                "candidate_source": candidate_source,
                "gpt_request_count": gpt_request_count,
                "stage_latencies_ms": timings,
                "requested_quantity": request.quantity,
                "returned_count": len(predictions),
                "adjustment_gates": adjustment_gates,
            },
        )

    def _can_use_yolo(self, request):
        return bool(
            self.use_yolo_first
            and self.yolo_backend is not None
            and self.yolo_backend.health().loaded
            and self.yolo_backend.supports_target(request.target_object)
        )

    def _quality_ground(self, *, request, image, trace, timings):
        pipeline_path = ["owlvit_initial"]
        gpt_request_count = 0

        initial_started = time.perf_counter()
        initial_candidates = self.owlvit_backend.detect_candidates(
            request,
            image=image,
            top_k=self.top_k_initial,
            thresholds=self.quality_thresholds,
            use_original_size=self.quality_use_original_image,
        )
        timings["owlvit_initial"] = self._elapsed(initial_started)
        initial_overlay = self._candidate_overlay(image, initial_candidates, {})
        trace.append(
            TraceEvent(
                stage="owlvit_initial_candidates",
                duration_ms=timings["owlvit_initial"],
                message=f"generated {len(initial_candidates)} quality candidates",
                data={
                    "candidate_count": len(initial_candidates),
                    "thresholds": self.quality_thresholds,
                    "original_size_inference": self.quality_use_original_image,
                },
            )
        )

        select_started = time.perf_counter()
        initial_decision = self._gpt_quality_selection(
            request,
            initial_overlay,
            initial_candidates,
            allow_region=True,
            stage="initial",
        )
        timings["gpt_initial_selection"] = self._elapsed(select_started)
        gpt_request_count += 1
        pipeline_path.append("gpt_initial_selection")
        trace.append(
            TraceEvent(
                stage="gpt_initial_selection",
                duration_ms=timings["gpt_initial_selection"],
                message=str(initial_decision.get("reason", "")),
                data=initial_decision,
            )
        )

        selected_initial = self._quality_candidate(initial_candidates, initial_decision)
        refinement_seed = (
            selected_initial.bbox_xyxy
            if selected_initial is not None
            else self._quality_region(initial_decision, image.width, image.height)
        )
        if refinement_seed is None:
            return GroundingResult.failure(
                request,
                status=GroundingStatus.CLARIFICATION_REQUIRED,
                backend_used=self.name,
                message="quality selection found neither a valid candidate nor a refinement region",
                clarification_required=True,
                trace=trace,
            )

        crop_box = self._expanded_crop(refinement_seed, image.width, image.height)
        crop = image.crop(tuple(int(round(value)) for value in crop_box.as_list()))
        refine_started = time.perf_counter()
        local_candidates = self.owlvit_backend.detect_candidates(
            request,
            image=crop,
            top_k=self.top_k_refined,
            thresholds=self.quality_thresholds,
            use_original_size=self.quality_use_original_image,
        )
        refined_candidates = [
            GroundingCandidate(
                candidate_id=f"refined_{index}",
                bbox_xyxy=BBoxXYXY(
                    x_min=item.bbox_xyxy.x_min + crop_box.x_min,
                    y_min=item.bbox_xyxy.y_min + crop_box.y_min,
                    x_max=item.bbox_xyxy.x_max + crop_box.x_min,
                    y_max=item.bbox_xyxy.y_max + crop_box.y_min,
                ).clipped(image.width, image.height),
                confidence=item.confidence,
                label=item.label,
                source=item.source,
                metadata={**item.metadata, "quality_crop": crop_box.as_list()},
            )
            for index, item in enumerate(local_candidates, start=1)
        ]
        timings["owlvit_local_refinement"] = self._elapsed(refine_started)
        pipeline_path.append("owlvit_local_refinement")
        trace.append(
            TraceEvent(
                stage="owlvit_local_refinement",
                duration_ms=timings["owlvit_local_refinement"],
                message=f"generated {len(refined_candidates)} refined candidates",
                data={"crop": crop_box.as_list()},
            )
        )

        selected = selected_initial
        selection_decision = initial_decision
        if refined_candidates:
            refined_overlay = self._candidate_overlay(image, refined_candidates, {})
            refined_started = time.perf_counter()
            refined_decision = self._gpt_quality_selection(
                request,
                refined_overlay,
                refined_candidates,
                allow_region=False,
                stage="refined",
            )
            timings["gpt_refined_selection"] = self._elapsed(refined_started)
            gpt_request_count += 1
            pipeline_path.append("gpt_refined_selection")
            trace.append(
                TraceEvent(
                    stage="gpt_refined_selection",
                    duration_ms=timings["gpt_refined_selection"],
                    message=str(refined_decision.get("reason", "")),
                    data=refined_decision,
                )
            )
            refined_selected = self._quality_candidate(refined_candidates, refined_decision)
            if refined_selected is not None:
                selected = refined_selected
                selection_decision = refined_decision

        if selected is None:
            return GroundingResult.failure(
                request,
                status=GroundingStatus.CLARIFICATION_REQUIRED,
                backend_used=self.name,
                message="quality refinement produced no selectable target candidate",
                clarification_required=True,
                trace=trace,
            )

        edge_image = self._edge_review_image(image, selected.bbox_xyxy)
        edge_started = time.perf_counter()
        edge_decision = self._gpt_quality_edge_review(request, edge_image)
        timings["gpt_edge_review"] = self._elapsed(edge_started)
        gpt_request_count += 1
        pipeline_path.append("gpt_edge_review")
        final_box, adjustment_applied, adjustment_gate = self._apply_safe_adjustment(
            selected.bbox_xyxy,
            edge_decision,
            image.width,
            image.height,
        )
        trace.append(
            TraceEvent(
                stage="gpt_edge_review",
                duration_ms=timings["gpt_edge_review"],
                message=adjustment_gate,
                data={**edge_decision, "adjustment_applied": adjustment_applied},
            )
        )

        confidence = float(
            selection_decision.get("confidence", selected.confidence)
            or selected.confidence
        )
        relation_match = selection_decision.get("relation_match")
        prediction = GroundingPrediction(
            bbox_xyxy=final_box,
            confidence=max(0.0, min(1.0, confidence)),
            label=selected.label,
            relation_match=relation_match,
            candidate_id=selected.candidate_id,
            metadata={"source": selected.source},
        )
        final_overlay = self._prediction_overlay(image, [prediction])
        self._save_debug(request, initial_overlay, final_overlay)
        return GroundingResult(
            request_id=request.request_id,
            status=GroundingStatus.SUCCESS,
            bbox_xyxy=prediction.bbox_xyxy,
            predictions=[prediction],
            confidence=prediction.confidence,
            relation_match=relation_match,
            backend_used=self.name,
            candidates=refined_candidates or initial_candidates,
            trace=trace,
            metadata={
                "performance_profile": "quality",
                "pipeline_path": pipeline_path,
                "gpt_request_count": gpt_request_count,
                "stage_latencies_ms": timings,
                "initial_candidate_count": len(initial_candidates),
                "refined_candidate_count": len(refined_candidates),
                "adjustment_applied": adjustment_applied,
                "adjustment_gate": adjustment_gate,
            },
        )

    def _local_fast_selection(self, request, candidates, ranked):
        no_attributes = not request.attributes
        plural = request.quantity in {QuantityIntent.MULTIPLE, QuantityIntent.ALL}

        if request.relations and ranked and no_attributes:
            if plural:
                selected = [item[2] for item in ranked if item[1] >= self.local_geometry_min_score]
                if len(selected) >= request.minimum_count:
                    return selected[:request.maximum_results]
            elif clear_winner(
                ranked,
                minimum_score=self.local_geometry_min_score,
                minimum_margin=self.local_geometry_min_margin,
            ):
                return [ranked[0][2]]

        if not request.relations and no_attributes:
            if plural and len(candidates) >= request.minimum_count:
                return candidates[:request.maximum_results]
            if (
                self.skip_gpt_when_unambiguous
                and len(candidates) == 1
                and candidates[0].confidence >= self.single_candidate_confidence
            ):
                return [candidates[0]]
        return []

    def _success_result(
        self, *, request, image, selected, candidates, trace, timings,
        pipeline_path, gpt_request_count, decision_reason, relation_match,
    ):
        predictions = [GroundingPrediction(
            bbox_xyxy=c.bbox_xyxy.padded(self.default_box_padding, image.width, image.height),
            confidence=c.confidence,
            label=c.label,
            relation_match=relation_match,
            candidate_id=c.candidate_id,
            metadata={"source": c.source},
        ) for c in selected]
        trace.append(TraceEvent(
            stage="early_return", message=decision_reason,
            data={"returned_count": len(predictions)},
        ))
        return GroundingResult(
            request_id=request.request_id,
            status=GroundingStatus.SUCCESS,
            bbox_xyxy=predictions[0].bbox_xyxy,
            predictions=predictions,
            confidence=predictions[0].confidence,
            relation_match=relation_match,
            backend_used=self.name,
            candidates=candidates,
            trace=trace,
            metadata={
                "pipeline_path": pipeline_path,
                "gpt_request_count": gpt_request_count,
                "stage_latencies_ms": timings,
                "requested_quantity": request.quantity,
                "returned_count": len(predictions),
            },
        )

    def _gpt_quality_selection(
        self,
        request,
        image,
        candidates,
        *,
        allow_region,
        stage,
    ):
        lines = "\n".join(
            f"C{index}: label={candidate.label}, "
            f"detector_confidence={candidate.confidence:.4f}, "
            f"box={candidate.bbox_xyxy.as_list()}"
            for index, candidate in enumerate(candidates, start=1)
        ) or "No detector candidate is available."
        prompt = f"""Return exactly one JSON object and no markdown.

Instruction: {request.instruction}
Target phrase: {request.target_phrase or request.target_object or 'object'}
Target attributes: {json.dumps(request.attributes)}
Excluded attributes: {json.dumps(request.metadata.get('negated_attributes', []))}
Spatial constraints: {json.dumps([item.model_dump(mode='json') for item in request.relations])}
Selection stage: {stage}

The image may contain cyan candidates C1, C2, and so on.
{lines}

Select the candidate satisfying the complete instruction. Do not select an
anchor or a distinctive subsection of the target. If no candidate is suitable
and allow_region={str(bool(allow_region)).lower()}, return a coarse target region
in full-image pixel coordinates for another detector pass. Do not invent a
region when the target cannot be located.

Schema:
{{
  "status": "selected" or "refine" or "failed",
  "selected_candidate": integer or null,
  "region": [x_min, y_min, x_max, y_max] or null,
  "confidence": number from 0 to 1,
  "relation_match": true or false,
  "reason": "short explanation"
}}"""
        return self._vision_json(prompt, image)

    @staticmethod
    def _quality_candidate(candidates, decision):
        if decision.get("status") != "selected":
            return None
        value = decision.get("selected_candidate")
        if value is None and isinstance(decision.get("selected_candidates"), list):
            entries = decision["selected_candidates"]
            value = entries[0].get("candidate") if entries else None
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            return None
        return candidates[index] if 0 <= index < len(candidates) else None

    @staticmethod
    def _quality_region(decision, image_width, image_height):
        if decision.get("status") != "refine":
            return None
        region = decision.get("region")
        if not isinstance(region, list) or len(region) != 4:
            return None
        try:
            values = [float(value) for value in region]
        except (TypeError, ValueError):
            return None
        if max(abs(value) for value in values) <= 1.5:
            values = [
                values[0] * image_width,
                values[1] * image_height,
                values[2] * image_width,
                values[3] * image_height,
            ]
        try:
            return BBoxXYXY(
                x_min=values[0],
                y_min=values[1],
                x_max=values[2],
                y_max=values[3],
            ).clipped(image_width, image_height)
        except ValueError:
            return None

    def _gpt_quality_edge_review(self, request, image):
        prompt = f"""Return exactly one JSON object and no markdown.

Instruction: {request.instruction}
The cyan box already identifies the selected target. Review only whether its
four edges enclose the entire visible target without excessive background.
Never contract the box around a face, torso, handle, label, logo, or another
subsection. Use accept when no small correction is clearly justified.

Each shift is a fraction of the current box width or height and must be between
-{self.maximum_edge_adjustment:.2f} and {self.maximum_edge_adjustment:.2f}.

Schema:
{{
  "decision": "accept" or "adjust" or "uncertain",
  "confidence": number from 0 to 1,
  "left_shift": number,
  "top_shift": number,
  "right_shift": number,
  "bottom_shift": number,
  "reason": "short explanation"
}}"""
        decision = self._vision_json(prompt, image)
        if decision.get("decision") != "adjust":
            decision.update(
                {
                    "left_shift": 0.0,
                    "top_shift": 0.0,
                    "right_shift": 0.0,
                    "bottom_shift": 0.0,
                }
            )
        return decision

    @staticmethod
    def _edge_review_image(image, box):
        review = box.padded(0.35, image.width, image.height)
        crop = image.crop(tuple(int(round(value)) for value in review.as_list()))
        draw = ImageDraw.Draw(crop)
        relative = [
            box.x_min - review.x_min,
            box.y_min - review.y_min,
            box.x_max - review.x_min,
            box.y_max - review.y_min,
        ]
        draw.rectangle(relative, outline="cyan", width=6)
        return crop

    def _gpt_final_decision(self, request, image, candidates):
        lines = "\n".join(
            f"C{i}: label={c.label}, detector_confidence={c.confidence:.4f}, "
            f"box={c.bbox_xyxy.as_list()}"
            for i, c in enumerate(candidates, start=1)
        )
        constraints = [constraint.model_dump(mode="json") for constraint in request.relations]
        prompt = f"""Return exactly one JSON object and no markdown.

Instruction: {request.instruction}
Target phrase: {request.target_phrase or request.target_object or 'object'}
Normalized target: {request.target_object or 'object'}
Requested quantity: {request.quantity}
Minimum count: {request.minimum_count}
Attributes: {json.dumps(request.attributes)}
Spatial constraints: {json.dumps(constraints)}

The image contains cyan target candidates labelled C1, C2, ... .
Yellow A-labels are anchor objects and must not be selected.
{lines}

Select only candidates that satisfy the complete instruction. Preserve every
attribute, quantity requirement, and spatial constraint. Do not guess when no
candidate is valid. For each selected target, edge shifts may correct the box,
but the final box must enclose the entire visible object. Never contract around
only a face, torso, handle, label, logo, or other distinctive subsection.

Each shift is a fraction of the candidate width or height bounded between
-{self.maximum_edge_adjustment:.2f} and {self.maximum_edge_adjustment:.2f}.

Schema:
{{
  "status": "selected" or "failed",
  "selected_candidates": [
    {{
      "candidate": integer,
      "confidence": number from 0 to 1,
      "relation_match": true or false,
      "left_shift": number,
      "top_shift": number,
      "right_shift": number,
      "bottom_shift": number
    }}
  ],
  "reason": "short explanation"
}}"""
        return self._vision_json(prompt, image)

    def _vision_json(self, prompt, image):
        response = self._client_for_request().responses.create(
            model=self.openai_model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            image,
                            quality=self.gpt_jpeg_quality,
                            max_width=self.gpt_image_max_width,
                        ),
                        "detail": self.image_detail,
                    },
                ],
            }],
        )
        return self._extract_json(response.output_text)

    @staticmethod
    def _extract_json(text):
        text = (text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"GPT response contained no JSON: {text[:300]}")
            return json.loads(match.group(0))

    @staticmethod
    def _selected_candidates_from_decision(candidates, decision, maximum_results):
        if decision.get("status") != "selected":
            return []
        entries = decision.get("selected_candidates")
        if not isinstance(entries, list):
            single = decision.get("selected_candidate")
            entries = [{"candidate": single, "confidence": decision.get("confidence", 0.0)}]
        output = []
        used = set()
        for entry in entries:
            try:
                index = int(entry.get("candidate")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= index < len(candidates) and index not in used:
                output.append((candidates[index], entry))
                used.add(index)
            if len(output) >= maximum_results:
                break
        return output

    def _apply_safe_adjustment(self, box, decision, image_width, image_height):
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        if confidence < self.minimum_adjustment_confidence:
            return (
                box.padded(self.default_box_padding, image_width, image_height),
                False,
                "adjustment rejected: confidence gate; padded original retained",
            )

        def shift(name):
            value = float(decision.get(name, 0.0) or 0.0)
            value = max(-self.maximum_edge_adjustment, min(self.maximum_edge_adjustment, value))
            # Positive left/top and negative right/bottom values contract the box.
            if name in {"left_shift", "top_shift"}:
                value = min(value, self.maximum_edge_contraction)
            else:
                value = max(value, -self.maximum_edge_contraction)
            return value

        adjusted = BBoxXYXY(
            x_min=box.x_min + shift("left_shift") * box.width,
            y_min=box.y_min + shift("top_shift") * box.height,
            x_max=box.x_max + shift("right_shift") * box.width,
            y_max=box.y_max + shift("bottom_shift") * box.height,
        ).clipped(image_width, image_height)

        overlap = box_iou(box, adjusted)
        area_ratio = adjusted.area / max(1.0, box.area)
        if overlap < self.minimum_adjusted_box_overlap:
            return box.padded(self.default_box_padding, image_width, image_height), False, "adjustment rejected: overlap gate"
        if not self.minimum_adjusted_area_ratio <= area_ratio <= self.maximum_adjusted_area_ratio:
            return box.padded(self.default_box_padding, image_width, image_height), False, "adjustment rejected: coverage gate"
        return adjusted.padded(self.default_box_padding, image_width, image_height), True, "safe bounded adjustment accepted"

    def _local_refine_after_selection(self, request, image, selected_item):
        candidate, current_box, decision = selected_item
        crop_box = self._expanded_crop(current_box, image.width, image.height)
        crop = image.crop(tuple(int(x) for x in crop_box.as_list()))
        local = self.owlvit_backend.detect_candidates(
            request, image=crop, top_k=self.top_k_refined, max_image_width=None
        )
        if not local:
            return selected_item
        mapped = []
        for item in local:
            mapped_box = BBoxXYXY(
                x_min=item.bbox_xyxy.x_min + crop_box.x_min,
                y_min=item.bbox_xyxy.y_min + crop_box.y_min,
                x_max=item.bbox_xyxy.x_max + crop_box.x_min,
                y_max=item.bbox_xyxy.y_max + crop_box.y_min,
            ).clipped(image.width, image.height)
            mapped.append((box_iou(current_box, mapped_box), item.confidence, mapped_box))
        mapped.sort(reverse=True, key=lambda x: (x[0], x[1]))
        best_overlap, _, best_box = mapped[0]
        # Never allow local refinement to collapse around a small subsection.
        area_ratio = best_box.area / max(1.0, current_box.area)
        if best_overlap >= 0.45 and area_ratio >= self.minimum_adjusted_area_ratio:
            return candidate, best_box.padded(self.default_box_padding, image.width, image.height), decision
        return selected_item

    def _expanded_crop(self, box, image_width, image_height):
        return BBoxXYXY(
            x_min=box.x_min - box.width * self.local_crop_margin,
            y_min=box.y_min - box.height * self.local_crop_margin,
            x_max=box.x_max + box.width * self.local_crop_margin,
            y_max=box.y_max + box.height * self.local_crop_margin,
        ).clipped(image_width, image_height)

    @staticmethod
    def _candidate_overlay(image, candidates, anchors_by_name):
        output = image.convert("RGB").copy()
        draw = ImageDraw.Draw(output)
        for index, candidate in enumerate(candidates, start=1):
            box = candidate.bbox_xyxy
            draw.rectangle(box.as_list(), outline="cyan", width=5)
            draw.text((box.x_min + 5, box.y_min + 5), f"C{index}", fill="cyan")
        anchor_index = 1
        for anchor_name, anchors in anchors_by_name.items():
            for anchor in anchors:
                box = anchor.bbox_xyxy
                draw.rectangle(box.as_list(), outline="yellow", width=4)
                draw.text((box.x_min + 5, box.y_min + 5), f"A{anchor_index}:{anchor_name}", fill="yellow")
                anchor_index += 1
        return output

    @staticmethod
    def _prediction_overlay(image, predictions):
        output = image.convert("RGB").copy()
        draw = ImageDraw.Draw(output)
        for index, prediction in enumerate(predictions, start=1):
            draw.rectangle(prediction.bbox_xyxy.as_list(), outline="lime", width=6)
            draw.text(
                (prediction.bbox_xyxy.x_min + 5, prediction.bbox_xyxy.y_min + 5),
                f"P{index}", fill="lime",
            )
        return output

    def _save_debug(self, request, candidates, final):
        if self.debug_output_dir is None:
            return
        output = self.debug_output_dir / request.request_id
        output.mkdir(parents=True, exist_ok=True)
        candidates.save(output / "01_candidates.jpg")
        final.save(output / "02_final_prediction.jpg")

    @staticmethod
    def _elapsed(started):
        return (time.perf_counter() - started) * 1000.0
