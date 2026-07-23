from contextlib import nullcontext
from PIL import Image

from grounding.backends.owlvit_backend import OwlViTBackend
from grounding.schemas import GroundingRequest, ImagePayload


class Batch(dict):
    def to(self, device):
        return self


class Tensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value


class Processor:
    def __init__(self):
        self.post_calls = 0
    def __call__(self, **kwargs):
        return Batch()
    def post_process_grounded_object_detection(self, outputs, threshold, target_sizes, text_labels):
        self.post_calls += 1
        if threshold > 0.01:
            return [{"boxes": Tensor([]), "scores": Tensor([]), "text_labels": []}]
        return [{
            "boxes": Tensor([[10.0, 10.0, 30.0, 30.0]]),
            "scores": Tensor([0.7]),
            "text_labels": ["package"],
        }]


class Model:
    def __init__(self): self.calls = 0
    def __call__(self, **kwargs):
        self.calls += 1
        return object()


class Torch:
    @staticmethod
    def inference_mode(): return nullcontext()


def test_threshold_relaxation_reuses_one_model_forward_pass(tmp_path):
    backend = OwlViTBackend(
        model_path=tmp_path,
        thresholds=[0.05, 0.02, 0.01, 0.005],
        max_image_width=50,
    )
    backend._started = True
    backend._processor = Processor()
    backend._model = Model()
    backend._torch = Torch()
    backend._device = "cpu"
    request = GroundingRequest(
        image=ImagePayload(base64_data="aGVsbG8=", media_type="image/jpeg"),
        instruction="find the package",
        target_object="package",
    )
    candidates = backend.detect_candidates(request, image=Image.new("RGB", (100, 80)))
    assert backend._model.calls == 1
    assert backend._processor.post_calls == 3
    assert candidates[0].bbox_xyxy.as_list() == [20.0, 20.0, 60.0, 60.0]
