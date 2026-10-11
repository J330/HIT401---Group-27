# backend/detector/inference.py
# Sources: AnomalyDINO -- Damm et al. (2025), https://huggingface.co/facebook/dinov2-base
# Grad-CAM: Selvaraju et al. (2017), model-aware PyTorch implementation in cam.py

import os
import logging
from threading import RLock
import numpy as np
import torch
from PIL import Image
from django.conf import settings
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
import timm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from .cam import generate_gradcam

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


# Like the older project, load trained models on the FIRST SCAN, not when
# Django is imported by Gunicorn or collectstatic. A model-free image can serve
# the website, but it cannot give AI predictions until real .pt files exist.
classifier_model = None
classifier_processor = None
classifier_config = None
GATE_CHECKPOINT = "facebook/dinov2-base"
GATE_THRESHOLD = 0.30  # cosine distance -- calibrate with ml/predict_cli.py first
gate_processor = None
gate_model = None
gate_centroids = None
_MODEL_LOAD_LOCK = RLock()


def _ensure_models_loaded():
    global classifier_model, classifier_processor, classifier_config
    global gate_processor, gate_model, gate_centroids
    if classifier_model is not None and gate_model is not None:
        return
    with _MODEL_LOAD_LOCK:
        if classifier_model is not None and gate_model is not None:
            return
        # Check for LFS pointers BEFORE passing them to torch.load so failure
        # is actionable instead of an obscure invalid-deserialization error.
        paths = [
            os.path.join(ML_MODELS_DIR, 'gate_centroids.pt'),
            MODEL_REGISTRY[ACTIVE_MODEL_KEY]['weights'],
        ]
        for path in paths:
            if not os.path.isfile(path):
                raise RuntimeError(f"Trained model file missing: {path}. Obtain the real .pt weights via git lfs pull or Cloud Storage.")
            with open(path, 'rb') as fh:
                header = fh.read(128)
            if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise RuntimeError(f"Trained model file is a Git LFS pointer, not actual weights: {path}. Run git lfs pull locally or configure _MODEL_BUCKET in Cloud Build.")
        # Assign globals only after all loads succeed; retry on later scans if
        # the model files were absent or the load failed.
        loaded_clf, loaded_processor, loaded_config = _load_active_classifier()
        loaded_gate_processor = AutoImageProcessor.from_pretrained(GATE_CHECKPOINT)
        loaded_gate_model = AutoModel.from_pretrained(GATE_CHECKPOINT).eval().to(DEVICE)
        loaded_centroids = torch.load(paths[0], map_location=DEVICE, weights_only=True)
        if not isinstance(loaded_centroids, dict):
            raise RuntimeError('Gate checkpoint is not a dictionary of class centroids.')
        classifier_model = loaded_clf
        classifier_processor = loaded_processor
        classifier_config = loaded_config
        gate_processor = loaded_gate_processor
        gate_model = loaded_gate_model
        gate_centroids = loaded_centroids
        logger.info('Loaded classifier %s and DINOv2 gate on first scan', ACTIVE_MODEL_KEY)


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


def _classify(pil_image):
    if classifier_config["kind"] == "timm":
        # timm models expect a plain normalised tensor, not a HF processor
        tf = Compose([Resize(224, 224), Normalize(), ToTensorV2()])
        pixel_values = tf(image=np.array(pil_image))["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier_model(pixel_values)
    else:
        inputs = classifier_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            logits = classifier_model(pixel_values=pixel_values).logits

    probs = torch.softmax(logits, dim=1)[0]
    predicted_idx = int(probs.argmax())
    return CLASS_NAMES[predicted_idx], float(probs[predicted_idx]), pixel_values, predicted_idx


def _explain(pixel_values, predicted_idx):
    # Grad-CAM over the predicted class logit, not the probabilities or
    # Hugging Face ModelOutput object. See detector/cam.py for layer choices.
    return generate_gradcam(classifier_model, ACTIVE_MODEL_KEY, pixel_values, predicted_idx)


def predict_image(image_file):
    """
    The single entry point detector/views.py calls.
    image_file: a Django UploadedFile (already opened).
    Returns a plain dict ready to hand to the serializer/response.
    """
    # Do not load hundreds of MB of AI weights when the server starts.
    _ensure_models_loaded()
    pil_image = Image.open(image_file).convert("RGB")
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
        }

    # Hooks belong to a shared classifier: serialize all forward calls while
    # the Grad-CAM hook is registered so another scan cannot overwrite it.
    with _CLASSIFIER_LOCK:
        label, confidence, pixel_values, predicted_idx = _classify(pil_image)
        heatmap_error = None
        try:
            heatmap_array = _explain(pixel_values, predicted_idx)
        except Exception as exc:
            # Preserve the prediction, but NEVER silently swallow the actual CAM
            # error: Django logs and (in DEBUG mode) the result page report it.
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
    }
