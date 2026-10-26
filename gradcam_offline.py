"""Standalone Grad-CAM sample exporter using the exact LeafVision v12 model.

Example (from project backend folder):
 python offline_gradcam.py --image path/to/leaf.jpg --output gradcam_example.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from gradcam_utils import LeafVisionGradCAM


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Grad-CAM for LeafVision v12")
    parser.add_argument("--model", default="model/leafvision_model_v12.keras")
    parser.add_argument("--classes", default="model/class_names.json")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam_example.jpg")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    if isinstance(classes, dict):
        classes = [classes[str(i)] for i in range(len(classes))]

    image_bytes = Path(args.image).read_bytes()
    with Image.open(args.image) as raw:
        picture = raw.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    batch = np.expand_dims(np.asarray(picture, dtype=np.float32), axis=0)

    result = LeafVisionGradCAM(model).generate(batch, image_bytes)
    import base64
    Path(args.output).write_bytes(base64.b64decode(result.pop("overlay_base64")))
    result["class_name"] = classes[result["class_index"]]
    print(json.dumps({**result, "saved": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
