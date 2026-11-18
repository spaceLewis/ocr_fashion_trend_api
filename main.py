from io import BytesIO
from enum import Enum
from threading import Lock

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from segmentation_output import (
    render_segmentation_overlay,
    serialize_segmentation_results,
)
from transformers import pipeline

app = FastAPI(title="API SegFormer Clothes Locale")


class ProcessingQuality(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


PROCESSING_RESOLUTIONS = {
    ProcessingQuality.FAST: 384,
    ProcessingQuality.BALANCED: 512,
    ProcessingQuality.HIGH: 767,
}

# Chargement unique du modèle au démarrage
print("Chargement du modèle sayeed99/segformer_b3_clothes...")
segmenter = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")
print("Modèle prêt !")
segmenter_lock = Lock()


async def load_uploaded_image(file: UploadFile):
    image_bytes = await file.read()
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def run_segmentation(image: Image.Image, quality: ProcessingQuality):
    image_processor = segmenter.image_processor
    with segmenter_lock:
        original_size = image_processor.size
        resolution = PROCESSING_RESOLUTIONS[quality]
        try:
            image_processor.size = {"height": resolution, "width": resolution}
            return segmenter(image)
        finally:
            image_processor.size = original_size


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    quality: ProcessingQuality = Query(
        default=ProcessingQuality.HIGH,
        description="fast (384), balanced (512) ou high (767) pixels d'entrée.",
    ),
):
    image = await load_uploaded_image(file)

    results = run_segmentation(image, quality)

    return serialize_segmentation_results(results)


@app.post("/segment/image")
async def segment_image_overlay(
    file: UploadFile = File(...),
    quality: ProcessingQuality = Query(
        default=ProcessingQuality.HIGH,
        description="fast (384), balanced (512) ou high (767) pixels d'entrée.",
    ),
):
    image = await load_uploaded_image(file)
    results = run_segmentation(image, quality)
    overlay = render_segmentation_overlay(image, results)

    image_buffer = BytesIO()
    overlay.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": 'attachment; filename="segmentation-overlay.png"'
        },
    )
