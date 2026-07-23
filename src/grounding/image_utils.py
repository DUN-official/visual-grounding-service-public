from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .exceptions import ImageInputError

DEFAULT_MAX_IMAGE_BYTES = 30 * 1024 * 1024


def _resolve_path(path_text, allowed_roots=None):
    path = Path(path_text).expanduser().resolve()
    if allowed_roots:
        roots = [Path(root).expanduser().resolve() for root in allowed_roots]
        if not any(path == root or root in path.parents for root in roots):
            raise ImageInputError(f"image path is outside configured roots: {path}")
    if not path.is_file():
        raise ImageInputError(f"image file not found: {path}")
    return path


def payload_bytes(payload, *, allowed_roots=None, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    if payload.path:
        path = _resolve_path(payload.path, allowed_roots)
        if path.stat().st_size > max_bytes:
            raise ImageInputError("image exceeds configured maximum size")
        media_type = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(
            path.suffix.lower(), "image/jpeg"
        )
        return path.read_bytes(), media_type
    encoded = payload.base64_data or ""
    media_type = payload.media_type
    if encoded.startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
            media_type = header.split(";", 1)[0].replace("data:", "") or media_type
        except ValueError as exc:
            raise ImageInputError("malformed data URL") from exc
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError("invalid Base64 image") from exc
    if len(raw) > max_bytes:
        raise ImageInputError("image exceeds configured maximum size")
    return raw, media_type


def load_pil_image(payload, *, allowed_roots=None, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    raw, _ = payload_bytes(payload, allowed_roots=allowed_roots, max_bytes=max_bytes)
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError("image could not be decoded") from exc
    return image.convert("RGB")


def resize_for_inference(image, max_width: int | None):
    if not max_width or image.width <= max_width:
        return image, 1.0, 1.0
    ratio = max_width / float(image.width)
    resized = image.resize((max_width, max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
    return resized, image.width / resized.width, image.height / resized.height


def image_to_data_url(image, quality=80, max_width: int | None = None):
    image, _, _ = resize_for_inference(image.convert("RGB"), max_width)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
