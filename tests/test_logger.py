import json

from grounding.evaluation.logger import GroundingJSONLLogger
from grounding.schemas import BBoxXYXY, GroundingRequest, GroundingResult, GroundingStatus, ImagePayload


def test_logger_redacts_uploaded_image_bytes(tmp_path):
    request = GroundingRequest(
        image=ImagePayload(base64_data="aGVsbG8=", media_type="image/jpeg"),
        instruction="find the bag",
    )
    result = GroundingResult(
        request_id=request.request_id,
        status=GroundingStatus.SUCCESS,
        bbox_xyxy=BBoxXYXY(x_min=0, y_min=0, x_max=10, y_max=10),
    )
    path = tmp_path / "requests.jsonl"
    GroundingJSONLLogger(path).log(request, result)
    record = json.loads(path.read_text(encoding="utf-8"))
    image = record["request"]["image"]
    assert image["base64_data"] == "<redacted>"
    assert image["size_bytes"] == 5
    assert image["sha256"]
