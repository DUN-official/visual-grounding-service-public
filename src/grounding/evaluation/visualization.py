from PIL import ImageDraw

def draw_result(image, result, label=None, width=5):
    output = image.convert("RGB").copy()
    if result.bbox_xyxy is None:
        return output
    draw = ImageDraw.Draw(output)
    box = result.bbox_xyxy
    draw.rectangle(box.as_list(), outline="cyan", width=width)
    text = label or f"{result.backend_used}: {result.confidence or 0.0:.3f}"
    draw.text((box.x_min + 5, max(0, box.y_min - 18)), text, fill="cyan")
    return output
