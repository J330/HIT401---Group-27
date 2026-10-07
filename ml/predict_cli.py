# ml/predict_cli.py

import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)

# Suppress noisy HF/transformers warnings for cleaner CLI output
warnings.filterwarnings("ignore")

# ---- Config ----
CLASSIFIER_CHECKPOINT = "facebook/convnextv2-base-22k-224"
CLASSIFIER_WEIGHTS = "models/convnextv2base.pt"
GATE_WEIGHTS = "models/gate_centroids.pt"
GATE_CHECKPOINT = "facebook/dinov2-base"
CLASS_NAMES = ["black_sigatoka", "healthy"]
GATE_THRESHOLD = 0.60


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Classifier
    clf_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_CHECKPOINT)
    clf_model = AutoModelForImageClassification.from_pretrained(
        CLASSIFIER_CHECKPOINT,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
    )
    clf_model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    clf_model.to(device).eval()

    # Gate
    gate_processor = AutoImageProcessor.from_pretrained(GATE_CHECKPOINT)
    gate_model = AutoModel.from_pretrained(GATE_CHECKPOINT).to(device).eval()
    gate_centroids = torch.load(GATE_WEIGHTS, map_location=device)

    return device, clf_processor, clf_model, gate_processor, gate_model, gate_centroids


def predict(image_path):
    if not Path(image_path).exists():
        print(f"ERROR: file not found: {image_path}")
        sys.exit(1)

    (
        device,
        clf_processor,
        clf_model,
        gate_processor,
        gate_model,
        gate_centroids,
    ) = load_models()

    image = Image.open(image_path).convert("RGB")

    # ---- Stage 1: gate ----
    gate_inputs = gate_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = gate_model(
            pixel_values=gate_inputs["pixel_values"].to(device)
        ).last_hidden_state[:, 0][0]

    nearest_class = None
    nearest_distance = float("inf")
    for class_name, centroid in gate_centroids.items():
        sim = F.cosine_similarity(embedding, centroid.to(device), dim=0).item()
        distance = 1 - sim
        if distance < nearest_distance:
            nearest_class, nearest_distance = class_name, distance

    # ---- Stage 2: classify (only if gate passes) ----
    if nearest_distance > GATE_THRESHOLD:
        print(f"RESULT       : not_black_sigatoka")
        print(f"Reason       : out-of-distribution (nearest={nearest_class}, distance={nearest_distance:.3f})")
        print(f"Note         : This looks like a different-disease leaf or a non-banana photo.")
        return

    clf_inputs = clf_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = clf_model(pixel_values=clf_inputs["pixel_values"].to(device)).logits
        probs = F.softmax(logits, dim=1)[0]

    idx = int(probs.argmax())
    confidence = float(probs[idx])

    # ---- Clean output ----
    print(f"RESULT       : {CLASS_NAMES[idx]}")
    print(f"Confidence   : {confidence:.3f}")
    print(f"Gate check   : passed (distance={nearest_distance:.3f} to {nearest_class})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_cli.py <path-to-image>")
        sys.exit(1)
    predict(sys.argv[1])