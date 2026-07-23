from contextlib import nullcontext

from PIL import Image

from grounding.backends.owlvit_backend import OwlViTBackend
from grounding.schemas import GroundingRequest, ImagePayload


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _FakeProcessor:
    def __init__(self):
        self.text = None
        self.images = None
        self.post_text_labels = None

    def __call__(self, *, text, images, return_tensors):
        self.text = text
        self.images = images
        return _FakeBatch()

    def post_process_grounded_object_detection(
        self, outputs, threshold, target_sizes, text_labels
    ):
        self.post_text_labels = text_labels
        return [{
            "boxes": _FakeTensor([[1.0, 2.0, 30.0, 40.0]]),
            "scores": _FakeTensor([0.9]),
            "text_labels": ["bag"],
        }]


class _FakeModel:
    def __call__(self, **kwargs):
        return object()


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


def test_multiple_text_queries_are_batched_for_one_image(tmp_path):
    backend = OwlViTBackend(model_path=tmp_path)
    backend._started = True
    backend._processor = _FakeProcessor()
    backend._model = _FakeModel()
    backend._torch = _FakeTorch()
    backend._device = "cpu"

    request = GroundingRequest(
        image=ImagePayload(base64_data="aGVsbG8=", media_type="image/jpeg"),
        instruction="find the bag on the table",
        target_object="bag",
        location_hint="on the table",
    )
    image = Image.new("RGB", (100, 80))

    candidates = backend.detect_candidates(request, image=image)

    expected = [["bag", "bag on the table", "find the bag on the table"]]
    assert backend._processor.text == expected
    assert backend._processor.post_text_labels == expected
    assert len(candidates) == 1
