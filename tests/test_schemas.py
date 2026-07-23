import pytest
from pydantic import ValidationError
from grounding.schemas import BBoxXYXY, GroundingRequest, ImagePayload

def test_bbox_rejects_inverted_coordinates():
    with pytest.raises(ValidationError):
        BBoxXYXY(x_min=10, y_min=10, x_max=5, y_max=20)

def test_image_payload_requires_one_source():
    with pytest.raises(ValidationError):
        ImagePayload()
    with pytest.raises(ValidationError):
        ImagePayload(path="a.jpg", base64_data="abc")

def test_request_generates_id():
    request = GroundingRequest(
        image=ImagePayload(path="/tmp/image.jpg"),
        instruction="find the chair",
    )
    assert request.request_id
