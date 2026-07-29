from contextlib import nullcontext
from pathlib import Path
import time

from PIL import Image

from ..evaluation.metrics import box_iou
from ..exceptions import ModelProvisioningError
from ..image_utils import load_pil_image, resize_for_inference
from ..interface import GroundingBackend
from ..schemas import (
    BBoxXYXY, GroundingCandidate, GroundingPrediction, GroundingResult,
    GroundingStatus, TraceEvent,
)
from ..task_parser import normalize_text


class OwlViTBackend(GroundingBackend):
    def __init__(
        self, *, model_path, device="auto", thresholds=None, top_k=5,
        max_pairwise_iou=0.85, max_image_width=960, use_fp16=True,
        warmup_on_startup=True, prompt_aliases=None, allowed_image_roots=None,
        max_image_bytes=30 * 1024 * 1024,
    ):
        super().__init__("owlvit")
        self.model_path = Path(model_path)
        self.device_setting = device
        self.thresholds = thresholds or [0.05, 0.02, 0.01, 0.005]
        self.top_k = top_k
        self.max_pairwise_iou = max_pairwise_iou
        self.max_image_width = max_image_width
        self.use_fp16 = use_fp16
        self.warmup_on_startup = warmup_on_startup
        self.prompt_aliases = {
            normalize_text(key): list(values)
            for key, values in (prompt_aliases or {}).items()
        }
        self.allowed_image_roots = allowed_image_roots
        self.max_image_bytes = max_image_bytes
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "cpu"

    def startup(self):
        if not self.model_path.is_dir():
            raise ModelProvisioningError(
                f"OWL ViT local model directory is missing: {self.model_path}"
            )
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self._device = (
            "cuda" if self.device_setting == "auto" and torch.cuda.is_available()
            else "cpu" if self.device_setting == "auto"
            else self.device_setting
        )
        self._processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(self.model_path), local_files_only=True
        ).to(self._device)
        self._model.eval()
        self._torch = torch
        self._started = True
        warmup = "not requested"
        if self.warmup_on_startup:
            try:
                self.detect_candidates(
                    _WarmupRequest(), image=Image.new("RGB", (320, 240), "black"),
                    labels=["object"], top_k=1,
                )
                warmup = "complete"
            except Exception as exc:
                warmup = f"skipped ({type(exc).__name__})"
        precision = "fp16 autocast" if self._use_amp() else "fp32"
        self._health_detail = f"OWL ViT loaded on {self._device}; {precision}; warmup {warmup}"
        self._model_reference = str(self.model_path)

    def shutdown(self):
        self._processor = None
        self._model = None
        self._torch = None
        super().shutdown()

    def _ground_impl(self, request):
        started = time.perf_counter()
        image = load_pil_image(
            request.image,
            allowed_roots=self.allowed_image_roots,
            max_bytes=self.max_image_bytes,
        )
        decode_ms = (time.perf_counter() - started) * 1000.0
        inference_started = time.perf_counter()
        candidates = self.detect_candidates(request, image=image, top_k=request.maximum_results)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        if not candidates:
            return GroundingResult.failure(
                request, backend_used=self.name,
                message="OWL ViT produced no candidate",
                clarification_required=True,
                trace=[TraceEvent(stage="owlvit_candidate_generation", duration_ms=inference_ms)],
            )
        count = request.maximum_results if request.quantity.value in {"multiple", "all"} else 1
        selected = candidates[:count]
        predictions = [GroundingPrediction(
            bbox_xyxy=c.bbox_xyxy, confidence=c.confidence, label=c.label,
            candidate_id=c.candidate_id,
        ) for c in selected]
        return GroundingResult(
            request_id=request.request_id,
            status=GroundingStatus.SUCCESS,
            bbox_xyxy=predictions[0].bbox_xyxy,
            predictions=predictions,
            confidence=predictions[0].confidence,
            backend_used=self.name,
            candidates=candidates,
            trace=[
                TraceEvent(stage="image_decode", duration_ms=decode_ms),
                TraceEvent(
                    stage="owlvit_candidate_generation", duration_ms=inference_ms,
                    message="selected open-vocabulary candidate(s)",
                    data={"candidate_count": len(candidates), "max_image_width": self.max_image_width},
                ),
            ],
            metadata={"stage_latencies_ms": {"image_decode": decode_ms, "owlvit": inference_ms}},
        )

    def detect_candidates(
        self,
        request,
        *,
        image=None,
        labels=None,
        top_k=None,
        max_image_width=None,
        thresholds=None,
        use_original_size=False,
    ):
        if not self._started:
            raise RuntimeError("OWL ViT is not started")
        image = image or load_pil_image(
            request.image,
            allowed_roots=self.allowed_image_roots,
            max_bytes=self.max_image_bytes,
        )
        original_width, original_height = image.width, image.height
        inference_width = (
            None
            if use_original_size
            else self.max_image_width if max_image_width is None else max_image_width
        )
        inference_image, scale_x, scale_y = resize_for_inference(
            image, inference_width
        )
        labels = labels or self._labels_for_request(request)
        labels = [x for x in dict.fromkeys(x.strip() for x in labels) if x] or ["object"]
        top_k = top_k or self.top_k
        text_labels = [labels]

        # The model forward pass is independent of score threshold. Run it once,
        # then reuse the same outputs while relaxing only post-processing.
        with self._inference_lock:
            inputs = self._processor(
                text=text_labels, images=inference_image, return_tensors="pt"
            ).to(self._device)
            with self._torch.inference_mode(), self._autocast_context():
                outputs = self._model(**inputs)

        collected = []
        for threshold in thresholds or self.thresholds:
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                threshold=float(threshold),
                target_sizes=[(inference_image.height, inference_image.width)],
                text_labels=text_labels,
            )[0]
            boxes = processed["boxes"].detach().cpu().tolist()
            scores = processed["scores"].detach().cpu().tolist()
            output_labels = processed.get("text_labels", labels)
            for index, (coords, score, label) in enumerate(zip(boxes, scores, output_labels), start=1):
                collected.append(GroundingCandidate(
                    candidate_id=f"owl_{index}",
                    bbox_xyxy=BBoxXYXY(
                        x_min=float(coords[0]) * scale_x,
                        y_min=float(coords[1]) * scale_y,
                        x_max=float(coords[2]) * scale_x,
                        y_max=float(coords[3]) * scale_y,
                    ).clipped(original_width, original_height),
                    confidence=float(score),
                    label=str(label),
                    source=self.name,
                    metadata={
                        "threshold": float(threshold),
                        "inference_size": [inference_image.width, inference_image.height],
                        "original_size": [original_width, original_height],
                    },
                ))
            if collected:
                break

        collected.sort(key=lambda item: item.confidence, reverse=True)
        kept = []
        for candidate in collected:
            if any(box_iou(candidate.bbox_xyxy, existing.bbox_xyxy) > self.max_pairwise_iou for existing in kept):
                continue
            candidate.candidate_id = f"owl_{len(kept) + 1}"
            kept.append(candidate)
            if len(kept) >= top_k:
                break
        return kept

    def _use_amp(self):
        return bool(self.use_fp16 and str(self._device).startswith("cuda"))

    def _autocast_context(self):
        if not self._use_amp() or not hasattr(self._torch, "autocast"):
            return nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)

    def _labels_for_request(self, request):
        target = normalize_text(request.target_object) or "object"
        labels = [target]
        if request.target_phrase and normalize_text(request.target_phrase) != target:
            labels.append(request.target_phrase)
        if request.attributes:
            labels.append(" ".join([*request.attributes, target]))
        if request.location_hint:
            labels.append(f"{target} {request.location_hint}")
        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        labels.extend(metadata.get("candidate_prompts", []))
        for canonical, aliases in self.prompt_aliases.items():
            normalized_aliases = {normalize_text(item) for item in aliases}
            if target == canonical or target in normalized_aliases:
                labels.extend([canonical, *aliases])
        if normalize_text(request.instruction) not in [normalize_text(x) for x in labels]:
            labels.append(request.instruction)
        return list(
            dict.fromkeys(
                value.strip()
                for value in labels
                if isinstance(value, str) and value.strip()
            )
        )[:12]


class _WarmupRequest:
    target_object = "object"
    target_phrase = "object"
    attributes = []
    location_hint = None
    instruction = "object"
