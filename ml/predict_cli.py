# ml/predict_cli.py
# Terminal test script for the full pipeline (open-set gate + classifier),
# no Django needed -- the fastest way to check all four image types:
# healthy, black_sigatoka, other_disease, non_banana.
# Usage: python predict_cli.py <path-to-image>
import sys
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel

# Match these two lines to whichever model won in evaluate.py
CLASSIFIER_CHECKPOINT = "facebook/convnextv2-base-22k-224"
CLASSIFIER_WEIGHTS = "models/convnextv2base.pt"
CLASS_NAMES = ["healthy", "black_sigatoka"]
GATE_THRESHOLD = 0.60  # cosine distance -- calibrate against other_disease/non_banana images


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_CHECKPOINT)
    clf_model = AutoModelForImageClassification.from_pretrained(
        CLASSIFIER_CHECKPOINT, num_labels=len(CLASS_NAMES), ignore_mismatched_sizes=True
    )
    clf_model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    clf_model.to(device).eval()

    gate_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    gate_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
    gate_centroids = torch.load("models/gate_centroids.pt", map_location=device)
    return device, clf_processor, clf_model, gate_processor, gate_model, gate_centroids


def predict(image_path):
    device, clf_processor, clf_model, gate_processor, gate_model, gate_centroids = load_models()
    pil_image = Image.open(image_path).convert("RGB")

    # Step 1: the open-set gate -- is this even Healthy or Black Sigatoka?
    gate_inputs = gate_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        embedding = gate_model(
            pixel_values=gate_inputs["pixel_values"].to(device)
        ).last_hidden_state[:, 0][0]

    best_class, best_distance = None, float("inf")
    for class_name, centroid in gate_centroids.items():
        distance = 1 - torch.nn.functional.cosine_similarity(embedding, centroid, dim=0).item()
        if distance < best_distance:
            best_class, best_distance = class_name, distance

    if best_distance > GATE_THRESHOLD:
        print(f"RESULT: not_black_sigatoka  (nearest={best_class}, distance={best_distance:.3f})")
        print("-> This is what a different-disease leaf OR a non-banana photo both look like.")
        return

    # Step 2: in-distribution -- run the actual Healthy/Black-Sigatoka classifier
    clf_inputs = clf_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        logits = clf_model(pixel_values=clf_inputs["pixel_values"].to(device)).logits
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    print(f"RESULT: {CLASS_NAMES[idx]}  (confidence={float(probs[idx]):.3f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_cli.py <path-to-image>")
        sys.exit(1)
    predict(sys.argv[1])
