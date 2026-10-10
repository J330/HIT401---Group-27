# backend/detector/inference.py
# Sources: AnomalyDINO -- Damm et al. (2025), https://huggingface.co/facebook/dinov2-base
# Grad-CAM: Selvaraju et al. (2017), model-aware PyTorch implementation in cam.py

import os
import logging
from threading import RLock
import numpy as np
import torch
from PIL import Image, ImageOps
from django.conf import settings
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
import timm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from .cam import generate_gradcam
from .background_filter import make_background_filtered_view

logger = logging.getLogger(__name__)
_CLASSIFIER_LOCK = RLock()  # CAM forward hooks must not overlap requests
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ML_MODELS_DIR = os.path.join(settings.BASE_DIR.parent, "ml", "models")
CLASS_NAMES = ["black_sigatoka", "healthy"]

# --- Every trained closed-set candidate, keyed exactly like evaluate.py's
#     output -- this is what makes the active model swappable from the
#     backend without touching any code. ---
MODEL_REGISTRY = {
    "efficientnetv2s": {
        "kind": "timm",
        "checkpoint": "efficientnetv2_rw_s",
        "weights": os.path.join(ML_MODELS_DIR, "effnetv2s.pt"),
    },
    "convnextv2": {
        "kind": "hf",
        "checkpoint": "facebook/convnextv2-base-22k-224",
        "weights": os.path.join(ML_MODELS_DIR, "convnextv2base.pt"),
        "cam_layer": lambda m: [m.convnextv2.encoder.stages[-1]],
    },
    "swin": {
        "kind": "hf",
        "checkpoint": "microsoft/swin-base-patch4-window7-224",
        "weights": os.path.join(ML_MODELS_DIR, "swinbase.pt"),
        "cam_layer": lambda m: [m.swin.encoder.layers[-1]],
        "hierarchical": True,
    },
    "vit": {
        "kind": "hf",
        "checkpoint": "google/vit-base-patch16-224",
        "weights": os.path.join(ML_MODELS_DIR, "vitbase.pt"),
        "cam_layer": lambda m: [m.vit.encoder.layer[-1]],
        "hierarchical": True,
    },
}

# Set via the ACTIVE_MODEL environment variable (see backend/backend/settings.py).
# Change it there (or in the Render dashboard) and restart to switch models --
# nothing in this file needs editing.
ACTIVE_MODEL_KEY = settings.ACTIVE_MODEL


def _load_active_classifier():
    config = MODEL_REGISTRY[ACTIVE_MODEL_KEY]
    if config["kind"] == "timm":
        model = timm.create_model(config["checkpoint"], pretrained=False, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(config["weights"], map_location=DEVICE))
        processor = None
    else:
        processor = AutoImageProcessor.from_pretrained(config["checkpoint"])
        model = AutoModelForImageClassification.from_pretrained(
            config["checkpoint"], num_labels=len(CLASS_NAMES), ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(config["weights"], map_location=DEVICE))
    model.to(DEVICE).eval()
    model.requires_grad_(False)  # inference only; input-gradient CAM still works
    return model, processor, config


classifier_model, classifier_processor, classifier_config = _load_active_classifier()
print(f"[inference.py] Active website model: {ACTIVE_MODEL_KEY}")

# --- DINOv2 open-set gate -- always loaded, regardless of active classifier ---
GATE_CHECKPOINT = "facebook/dinov2-base"
GATE_THRESHOLD = 0.30  # cosine distance -- calibrate with ml/predict_cli.py first
gate_processor = AutoImageProcessor.from_pretrained(GATE_CHECKPOINT)
gate_model = AutoModel.from_pretrained(GATE_CHECKPOINT).eval().to(DEVICE)
gate_centroids = torch.load(os.path.join(ML_MODELS_DIR, "gate_centroids.pt"), map_location=DEVICE)


def _gate_check(pil_image):
    # Returns (in_distribution, nearest_class, distance)
    inputs = gate_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        embedding = gate_model(pixel_values=inputs["pixel_values"].to(DEVICE)).last_hidden_state[:, 0][0]

    best_class, best_distance = None, float("inf")
    for class_name, centroid in gate_centroids.items():
        distance = 1 - torch.nn.functional.cosine_similarity(embedding, centroid, dim=0).item()
        if distance < best_distance:
            best_class, best_distance = class_name, distance
    return best_distance <= GATE_THRESHOLD, best_class, best_distance


def _prepare_classifier_input(pil_image):
    """Apply the same tensor preprocessing for prediction and explanation."""
    if classifier_config["kind"] == "timm":
        tf = Compose([Resize(224, 224), Normalize(), ToTensorV2()])
        return tf(image=np.array(pil_image))["image"].unsqueeze(0).to(DEVICE)
    inputs = classifier_processor(images=pil_image, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE)


def _classify(pil_image):
    pixel_values = _prepare_classifier_input(pil_image)
    with torch.no_grad():
        if classifier_config["kind"] == "timm":
            logits = classifier_model(pixel_values)
        else:
            logits = classifier_model(pixel_values=pixel_values).logits

    probs = torch.softmax(logits, dim=1)[0]
    predicted_idx = int(probs.argmax())
    return CLASS_NAMES[predicted_idx], float(probs[predicted_idx]), pixel_values, predicted_idx


def _explain(pixel_values, predicted_idx):
    # Grad-CAM over the predicted class logit, not the probabilities or
    # Hugging Face ModelOutput object. See detector/cam.py for layer choices.
    return generate_gradcam(classifier_model, ACTIVE_MODEL_KEY, pixel_values, predicted_idx)


def predict_image(image_file):
    """Stage 1 and 2 use the original; isolation applies only to Grad-CAM."""
    pil_image = ImageOps.exif_transpose(Image.open(image_file)).convert("RGB")
    # Stage 1 checks the ORIGINAL image to maintain the gate calibration.
    in_distribution, nearest_class, distance = _gate_check(pil_image)

    if not in_distribution:
        return {
            "label": "not_black_sigatoka",
            "confidence": round(1 - distance, 4),
            "is_known_disease": False,
            "is_banana_leaf": None,  # gate doesn't distinguish which negative case
            "heatmap_array": None,
            "gate_nearest_class": nearest_class,
            "gate_distance": round(float(distance), 5),
            "gate_threshold": GATE_THRESHOLD,
            "classifier_key": ACTIVE_MODEL_KEY,
            "background_removed_applied": False,
            "background_removal_method": None,
            "background_removal_reason": "Not applied because Stage 1 rejected the image.",
            "heatmap_source_image": None,
            "heatmap_foreground_mask": None,
        }

    # Stage 2 prediction ALWAYS uses the original, unchanged photograph.
    # The U2Net cutout is made only after a class has been selected, and is
    # used for a separate Grad-CAM forward pass targeting THAT original class.
    # Because this second pass sees a different input, its attribution is for
    # the isolated view; it must not be described as attribution of the original
    # forward pass. The predicted class and confidence remain unchanged.
    # Keep the first, unchanged image inference isolated from concurrent CAM
    # hooks, but do not hold the classifier lock while U2Net downloads/executes.
    with _CLASSIFIER_LOCK:
        label, confidence, original_pixels, predicted_idx = _classify(pil_image)

    filtered = make_background_filtered_view(pil_image)
    cam_image = filtered.image if filtered.applied else pil_image
    cam_pixels = (_prepare_classifier_input(cam_image)
                  if filtered.applied else original_pixels)

    heatmap_error = None
    try:
        with _CLASSIFIER_LOCK:
            heatmap_array = _explain(cam_pixels, predicted_idx)
    except Exception as exc:
        logger.exception("Grad-CAM failed for active model %s", ACTIVE_MODEL_KEY)
        heatmap_array = None
        heatmap_error = (
            f"Grad-CAM ({ACTIVE_MODEL_KEY}): {type(exc).__name__}: {exc}"
            if settings.DEBUG else "Grad-CAM could not be generated for this scan."
        )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "is_known_disease": True,
        "is_banana_leaf": True,
        "heatmap_array": heatmap_array,
        "heatmap_error": heatmap_error,
        "gate_nearest_class": nearest_class,
        "gate_distance": round(float(distance), 5),
        "gate_threshold": GATE_THRESHOLD,
        "classifier_key": ACTIVE_MODEL_KEY,
        "background_removed_applied": bool(filtered.applied),
        "background_removal_method": filtered.method,
        "background_removal_reason": filtered.reason,
        "heatmap_source_image": cam_image,
        "heatmap_foreground_mask": filtered.foreground_mask if filtered.applied else None,
    }
