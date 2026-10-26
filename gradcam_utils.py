"""Grad-CAM for the exact nested MobileNetV2 backbone in LeafVision v12.

Does not alter model weights. The final backbone activation ('out_relu') is
connected through the original trained GAP and classifier to obtain logit
scores, avoiding near-zero softmax gradients for confident predictions.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image


class LeafVisionGradCAM:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.augmentation = model.get_layer("training_augmentation")
        self.rescaling = model.get_layer("mobilenet_preprocessing")
        backbone = model.get_layer("mobilenetv2_1.00_224")
        self.feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer("out_relu").output,
            name="leafvision_gradcam_features",
        )
        self.pooling = model.get_layer("global_average_pooling2d")
        self.classifier = model.get_layer("plant_prediction")

        # v12: backbone -> GAP -> Dropout(0.3; identity at inference) ->
        # Dense(12, softmax). We use pre-softmax logits for Grad-CAM.
        assert getattr(self.classifier.activation, "__name__", "") == "softmax", (
            "Unexpected classifier: check the v12 architecture before using Grad-CAM"
        )

    def generate(self, batch: np.ndarray, original_image_bytes: bytes) -> dict[str, Any]:
        if batch.shape != (1, 224, 224, 3):
            raise ValueError(f"Expected RGB batch (1,224,224,3), got {batch.shape}")

        x = tf.convert_to_tensor(batch, dtype=tf.float32)
        x = self.augmentation(x, training=False)
        x = self.rescaling(x)

        with tf.GradientTape() as tape:
            fmap = self.feature_model(x, training=False)
            pooled = self.pooling(fmap)
            # In v12 Dropout is disabled during inference; its output == pooled.
            logits = tf.linalg.matmul(pooled, self.classifier.kernel)
            if self.classifier.bias is not None:
                logits = logits + self.classifier.bias
            probs = tf.nn.softmax(logits, axis=-1)
            class_index = int(tf.argmax(probs[0]).numpy())
            class_logit = logits[:, class_index]

        gradients = tape.gradient(class_logit, fmap)
        if gradients is None:
            raise RuntimeError("No gradient found for Grad-CAM feature map")

        # Fail closed if reconstruction differs from the original model.
        # This helps detect unanticipated edits to the v12 architecture.
        reference = self.model(tf.convert_to_tensor(batch), training=False).numpy()[0]
        reconstructed = probs.numpy()[0]
        if not np.allclose(reference, reconstructed, rtol=1e-3, atol=1e-4):
            raise RuntimeError("Grad-CAM reconstructed scores differ from v12 inference")

        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        activation_map = tf.reduce_sum(fmap * weights, axis=-1)[0]
        activation_map = tf.nn.relu(activation_map)
        activation_map = activation_map.numpy().astype(np.float32)
        heatmax = float(np.max(activation_map))
        no_signal = heatmax < 1e-12
        heatmap = activation_map / heatmax if not no_signal else np.zeros_like(activation_map)

        with Image.open(io.BytesIO(original_image_bytes)) as img:
            original = img.convert("RGB")
        original.thumbnail((720, 720), Image.Resampling.BILINEAR)
        overlay = overlay_gradcam(original, heatmap)
        output = io.BytesIO()
        overlay.save(output, format="JPEG", quality=85, optimize=True)

        return {
            "class_index": class_index,
            "confidence": round(float(reference[class_index]) * 100, 2),
            "feature_layer": "mobilenetv2_1.00_224/out_relu",
            "overlay_base64": base64.b64encode(output.getvalue()).decode("ascii"),
            "no_signal": no_signal,
        }


def overlay_gradcam(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """Blend a red-yellow hotspot into the original photo (no false blue regions)."""
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
    resized = resized.resize(original.size, Image.Resampling.BICUBIC)
    strength = np.asarray(resized, dtype=np.float32) / 255.0
    strength = np.clip(strength, 0.0, 1.0)
    orig = np.asarray(original, dtype=np.float32)
    yellow = np.empty_like(orig)
    yellow[..., 0] = 255.0
    yellow[..., 1] = np.maximum(0.0, 255.0 - 200.0 * strength)
    yellow[..., 2] = 0.0
    alpha = (0.68 * strength)[..., np.newaxis]
    result = (orig * (1.0 - alpha) + yellow * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
