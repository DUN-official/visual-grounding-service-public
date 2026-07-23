import inspect

from grounding.backends.gpt_guided_owlvit_backend import GPTGuidedOWLViTBackend
from grounding.backends.owlvit_backend import OwlViTBackend
from grounding.backends.yolo_backend import YoloBackend

FORBIDDEN = (
    "snapshot_download",
    "hf_hub_download",
    "from_pretrained(",
    "urlretrieve(",
)


def test_ground_methods_do_not_download_weights():
    for backend_class in [YoloBackend, OwlViTBackend, GPTGuidedOWLViTBackend]:
        source = inspect.getsource(backend_class._ground_impl)
        for forbidden in FORBIDDEN:
            assert forbidden not in source
