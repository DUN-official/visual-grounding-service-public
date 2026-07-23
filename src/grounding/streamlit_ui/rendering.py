"""Rendering helpers shared by Streamlit views."""

from __future__ import annotations

from io import BytesIO
import json

from PIL import Image, ImageDraw, ImageFont

from ..schemas import GroundingResult


BOX_COLOR = (255, 176, 0, 255)
MASK_COLOR = (0, 180, 216, 82)


def annotate_image(image: Image.Image, result: GroundingResult) -> Image.Image:
    canvas = image.convert("RGBA")
    mask_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)

    for prediction in result.predictions:
        segmentation = prediction.metadata.get("segmentation")
        if not isinstance(segmentation, dict):
            continue
        for contour in segmentation.get("contours") or []:
            points = [tuple(point) for point in contour if len(point) == 2]
            if len(points) >= 3:
                mask_draw.polygon(points, fill=MASK_COLOR)

    fallback_segmentation = result.metadata.get("segmentation")
    if isinstance(fallback_segmentation, dict):
        for contour in fallback_segmentation.get("contours") or []:
            points = [tuple(point) for point in contour if len(point) == 2]
            if len(points) >= 3:
                mask_draw.polygon(points, fill=MASK_COLOR)

    canvas = Image.alpha_composite(canvas, mask_layer)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    predictions = result.predictions
    if not predictions and result.bbox_xyxy is not None:
        predictions = []

    if predictions:
        boxes = [
            (prediction.bbox_xyxy, prediction.label, prediction.confidence)
            for prediction in predictions
        ]
    elif result.bbox_xyxy is not None:
        boxes = [(result.bbox_xyxy, result.backend_used, result.confidence or 0.0)]
    else:
        boxes = []

    for box, label, confidence in boxes:
        coordinates = [box.x_min, box.y_min, box.x_max, box.y_max]
        draw.rectangle(coordinates, outline=BOX_COLOR, width=4)
        caption = f"{label or 'target'} {confidence:.2f}"
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = max(0, int(box.x_min))
        y = max(0, int(box.y_min) - text_height - 8)
        draw.rectangle(
            (x, y, x + text_width + 10, y + text_height + 6),
            fill=(18, 18, 18, 220),
        )
        draw.text((x + 5, y + 3), caption, fill=BOX_COLOR, font=font)
    return canvas.convert("RGB")


def image_bytes(image: Image.Image, *, format_name: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def result_json(result: GroundingResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2)
