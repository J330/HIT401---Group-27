# backend/detector/inference.py
# Loads the DINOv2 open-set gate once at Django startup, plus whichever
# closed-set classifier settings.ACTIVE_MODEL names, and exposes a single
# predict_image() function for views.py to call per request.
#
# Sources: AnomalyDINO -- Damm et al. (2025), https://huggingface.co/facebook/dinov2-base
#          jacobgil/pytorch-grad-cam -- https://github.com/jacobgil/pytorch-grad-cam
import os
import numpy as np
import torch
from PIL import Image
from django.conf import settings
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
import timm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
    if classifier_config["kind"] == "timm":
        cam = GradCAM(model=classifier_model, target_layers=[classifier_model.conv_head])
    elif classifier_config.get("hierarchical"):
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
            return result.transpose(2, 3).transpose(1, 2)

        cam = GradCAM(
            model=classifier_model,
            target_layers=classifier_config["cam_layer"](classifier_model),
            reshape_transform=reshape_transform,
        )
    else:
        cam = GradCAM(model=classifier_model, target_layers=classifier_config["cam_layer"](classifier_model))

    grayscale_cam = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(predicted_idx)])
    return grayscale_cam[0]


def predict_image(image_file):
    """
    The single entry point detector/views.py calls.
    image_file: a Django UploadedFile (already opened).
    Returns a plain dict ready to hand to the serializer/response.
    """
    pil_image = Image.open(image_file).convert("RGB")
    in_distribution, nearest_class, distance = _gate_check(pil_image)

    if not in_distribution:
        return {
            "label": "not_black_sigatoka",
            "confidence": round(1 - distance, 4),
            "is_known_disease": False,
            "is_banana_leaf": None,  # gate doesn't distinguish which negative case
            "heatmap_array": None,
        }

    label, confidence, pixel_values, predicted_idx = _classify(pil_image)
    try:
        heatmap_array = _explain(pixel_values, predicted_idx)
    except Exception:
        heatmap_array = None  # never let explainability break a prediction

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "is_known_disease": True,
        "is_banana_leaf": True,
        "heatmap_array": heatmap_array,
    }
