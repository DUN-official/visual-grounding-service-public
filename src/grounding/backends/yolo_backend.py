from pathlib import Path
import time

from PIL import Image

from ..exceptions import ModelProvisioningError
from ..image_utils import load_pil_image
from ..interface import GroundingBackend
from ..schemas import (
    BBoxXYXY, GroundingCandidate, GroundingPrediction, GroundingResult,
    GroundingStatus, TraceEvent,
)
from ..task_parser import infer_target_from_vocabulary, normalize_text
from ..ultralytics_env import configure_ultralytics_directory


class YoloBackend(GroundingBackend):
    def __init__(
        self, *, weights_path, class_aliases, confidence_threshold=0.25,
        image_size=640, device="auto", max_candidates=20,
        warmup_on_startup=True, allowed_image_roots=None,
        max_image_bytes=30 * 1024 * 1024,
    ):
        super().__init__("yolo")
        self.weights_path = Path(weights_path)
        self.class_aliases = class_aliases
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.max_candidates = max_candidates
        self.warmup_on_startup = warmup_on_startup
        self.allowed_image_roots = allowed_image_roots
        self.max_image_bytes = max_image_bytes
        self._model = None

    def startup(self):
        if not self.weights_path.is_file():
            raise ModelProvisioningError(
                f"YOLO weights must exist before startup: {self.weights_path}"
            )
        configure_ultralytics_directory()
        from ultralytics import YOLO
        self._model = YOLO(str(self.weights_path))
        warmup = "not requested"
        if self.warmup_on_startup:
            try:
                self._predict(Image.new("RGB", (64, 64), "black"))
                warmup = "complete"
            except Exception as exc:
                warmup = f"skipped ({type(exc).__name__})"
        self._started = True
        self._health_detail = f"YOLO loaded from local weights; warmup {warmup}"
        self._model_reference = str(self.weights_path)

    def shutdown(self):
        self._model = None
        super().shutdown()

    def _canonical_classes(self, target):
        target = normalize_text(target)
        matches = []
        for canonical, aliases in self.class_aliases.items():
            terms = {normalize_text(canonical), *{normalize_text(x) for x in aliases}}
            if target in terms:
                matches.append(canonical)
        return matches

    def supports_target(self, target):
        return bool(self._canonical_classes(target))

    def supports(self, request):
        return self.supports_target(request.target_object)

    def detect_scene(self, image):
        """Run YOLO once and return all detections for target/anchor filtering."""
        if not self._started:
            raise RuntimeError("YOLO is not started")
        with self._inference_lock:
            results = self._predict(image)
        if not results:
            return []
        result = results[0]
        candidates = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for index, (coords, score, class_id) in enumerate(zip(xyxy, scores, classes), start=1):
                class_name = str(result.names[int(class_id)])
                candidates.append(GroundingCandidate(
                    candidate_id=f"yolo_scene_{index}",
                    bbox_xyxy=BBoxXYXY(
                        x_min=float(coords[0]), y_min=float(coords[1]),
                        x_max=float(coords[2]), y_max=float(coords[3]),
                    ).clipped(image.width, image.height),
                    confidence=float(score),
                    label=class_name,
                    source=self.name,
                ))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[:self.max_candidates]

    def filter_candidates(self, scene_candidates, target, *, top_k=None):
        canonicals = {normalize_text(x) for x in self._canonical_classes(target)}
        if not canonicals:
            return []
        output = [
            candidate for candidate in scene_candidates
            if normalize_text(candidate.label) in canonicals
        ]
        output.sort(key=lambda item: item.confidence, reverse=True)
        output = output[: top_k or self.max_candidates]
        for index, candidate in enumerate(output, start=1):
            candidate.candidate_id = f"yolo_{index}"
        return output

    def detect_candidates(self, request, *, image=None, target=None, top_k=None):
        image = image or load_pil_image(
            request.image,
            allowed_roots=self.allowed_image_roots,
            max_bytes=self.max_image_bytes,
        )
        target = target or request.target_object
        if not target:
            vocabulary = [
                term for canonical, aliases in self.class_aliases.items()
                for term in [canonical, *aliases]
            ]
            target = infer_target_from_vocabulary(request.instruction, vocabulary)
        return self.filter_candidates(self.detect_scene(image), target, top_k=top_k)

    def _predict(self, image):
        kwargs = {
            "source": image,
            "conf": self.confidence_threshold,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device != "auto":
            kwargs["device"] = self.device
        return self._model.predict(**kwargs)

    def _ground_impl(self, request):
        started = time.perf_counter()
        image = load_pil_image(
            request.image,
            allowed_roots=self.allowed_image_roots,
            max_bytes=self.max_image_bytes,
        )
        load_ms = (time.perf_counter() - started) * 1000.0
        target = request.target_object
        if not self.supports_target(target):
            return GroundingResult.failure(
                request,
                status=GroundingStatus.INVALID_REQUEST,
                backend_used=self.name,
                message=f"unsupported YOLO target: {target!r}",
                clarification_required=True,
            )
        inference_started = time.perf_counter()
        candidates = self.detect_candidates(
            request, image=image, target=target, top_k=request.maximum_results
        )
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        if not candidates:
            return GroundingResult.failure(
                request, backend_used=self.name,
                message=f"no {target!r} detection above threshold",
                clarification_required=True,
                trace=[TraceEvent(stage="yolo_detection", duration_ms=inference_ms)],
            )
        count = request.maximum_results if request.quantity.value in {"multiple", "all"} else 1
        selected = candidates[:count]
        predictions = [GroundingPrediction(
            bbox_xyxy=c.bbox_xyxy,
            confidence=c.confidence,
            label=c.label,
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
                TraceEvent(stage="image_decode", duration_ms=load_ms),
                TraceEvent(
                    stage="yolo_detection", duration_ms=inference_ms,
                    message=f"returned {len(predictions)} target prediction(s)",
                    data={"candidate_count": len(candidates), "image_size": self.image_size},
                ),
            ],
            metadata={"stage_latencies_ms": {"image_decode": load_ms, "yolo": inference_ms}},
        )
