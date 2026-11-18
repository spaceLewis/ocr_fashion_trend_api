from pathlib import Path

from PIL import Image
from transformers import pipeline

from segmentation_output import render_segmentation_overlay


input_path = Path("test.png")
with Image.open(input_path) as source_image:
    image = source_image.convert("RGB")

segmenter = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")
results = segmenter(image)
overlay = render_segmentation_overlay(image, results)

output_path = Path("resultats_overlay.png")
overlay.save(output_path, format="PNG")
print(f"Image enregistrée dans {output_path.resolve()}")
