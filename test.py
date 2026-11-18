import json
from pathlib import Path

from transformers import pipeline

from segmentation_output import serialize_segmentation_results

# Télécharge automatiquement le modèle déjà entraîné (1 seule fois)
segmenter = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")

# Analyse votre image et renvoie les mêmes masques que l'API
resultats = segmenter("test.png")

output_path = Path("resultats.json")
output_path.write_text(
    json.dumps(serialize_segmentation_results(resultats), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Résultats enregistrés dans {output_path.resolve()}")