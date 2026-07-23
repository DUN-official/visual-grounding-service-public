from PIL import Image

from grounding.schemas import (
    BBoxXYXY,
    GroundingPrediction,
    GroundingResult,
    GroundingStatus,
)
from grounding.streamlit_ui.rendering import annotate_image, image_bytes, result_json


def test_annotated_image_and_download_formats():
    result = GroundingResult(
        request_id="test",
        status=GroundingStatus.SUCCESS,
        predictions=[
            GroundingPrediction(
                bbox_xyxy=BBoxXYXY(x_min=10, y_min=12, x_max=60, y_max=70),
                confidence=0.9,
                label="package",
            )
        ],
        backend_used="owlvit",
    )
    image = Image.new("RGB", (100, 100), "white")

    rendered = annotate_image(image, result)

    assert rendered.size == image.size
    assert image_bytes(rendered).startswith(b"\x89PNG")
    assert '"backend_used": "owlvit"' in result_json(result)
