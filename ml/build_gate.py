# ml/build_gate.py
# https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_With_DINOv2_WACV_2025_paper.pdf
# Model: https://huggingface.co/facebook/dinov2-base

import os
import torch
from transformers import AutoModel, AutoImageProcessor
from common import get_loaders

CHECKPOINT = "facebook/dinov2-base"
OUT_PATH = "models/gate_centroids.pt"
EMBED_DIM = 768  # DINOv2-Base's embedding size


def get_embedding(model, pixel_values):
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    return out.last_hidden_state[:, 0]  # the CLS token = one vector per image


def build_centroids():
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT).eval()
    image_size = processor.crop_size["height"]

    train_loader, _, classes = get_loaders(
        "../data/processed", image_size, processor.image_mean, processor.image_std
    )

    sums = {c: torch.zeros(EMBED_DIM) for c in classes}
    counts = {c: 0 for c in classes}

    for images, labels in train_loader:
        embeddings = get_embedding(model, images)
        for emb, label in zip(embeddings, labels):
            class_name = classes[label]
            sums[class_name] += emb
            counts[class_name] += 1

    centroids = {c: sums[c] / counts[c] for c in classes}
    torch.save(centroids, OUT_PATH)
    print(f"Saved centroids for {list(classes)} to {OUT_PATH}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    build_centroids()
    print("Next: test predict_cli.py on data/raw/other_disease and data/raw/non_banana")
    print("images and adjust GATE_THRESHOLD in predict_cli.py / inference.py if needed.")
