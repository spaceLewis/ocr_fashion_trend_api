import base64
from io import BytesIO

from PIL import Image


OVERLAY_COLORS = (
    (239, 68, 68),
    (59, 130, 246),
    (34, 197, 94),
    (234, 179, 8),
    (168, 85, 247),
    (20, 184, 166),
)


def serialize_segmentation_results(results):
    serialized_results = []

    for result in results:
        mask_buffer = BytesIO()
        result["mask"].save(mask_buffer, format="PNG")
        serialized_results.append(
            {
                "label": result["label"],
                "score": result["score"],
                "mask": base64.b64encode(mask_buffer.getvalue()).decode("utf-8"),
            }
        )

    return serialized_results


def render_segmentation_overlay(image, results, opacity=128):
    overlay = image.convert("RGBA")

    for index, result in enumerate(results):
        mask = result["mask"].convert("L")
        if mask.size != overlay.size:
            mask = mask.resize(overlay.size, Image.Resampling.NEAREST)

        alpha = mask.point(lambda value: value * opacity // 255)
        color_layer = Image.new(
            "RGBA",
            overlay.size,
            OVERLAY_COLORS[index % len(OVERLAY_COLORS)] + (0,),
        )
        color_layer.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, color_layer)

    return overlay.convert("RGB")
