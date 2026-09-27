"""Two-stage inference for the banana leaf website.

Stage 1 uses the already-trained DINOv2 Base backbone as a binary visual
similarity gate with two reference groups only:

    Banana Leaf
    Not BananaLeaf

The binary reference bank is built from the user's existing dataset with
``build_initial_references.py``. Stage 1 does not use the five disease labels.

Stage 2 then performs the normal health classification. When DINOv2 Base is
selected, the five-class classifier output is additionally checked against
verified disease reference embeddings and can return Unknown instead of
forcing a disease label.

Required DINOv2 files:
    trainedai/dinov2-base/dinov2_banana_disease.pth
    trainedai/dinov2-base/dinov2_labels.json
    trainedai/dinov2-base/initial_references.pt
    trainedai/dinov2-base/disease_references.pt
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
from django.conf import settings
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from .background_filter import make_background_filtered_view
from .exceptions import InvalidImageError, ModelNotAvailableError, PredictionError
from .gradcam import generate_gradcam_data_url


MODEL_SPECS = {
    "dinov2-base": {
        "display_name": "DINOv2 Base",
        "type": "timm_dinov2",
        "folder": "dinov2-base",
        "description": "DINOv2 classifier with positive-reference similarity checking",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_SCAN_MODEL_KEY = "dinov2-base"

# Starting values only. If similarity_config.json exists next to the trained
# DINOv2 model, its values override these.
DEFAULT_SIMILARITY_THRESHOLD = 0.72
DEFAULT_INITIAL_BANANA_THRESHOLD = 0.65
DEFAULT_INITIAL_BINARY_MARGIN = 0.02
DEFAULT_SIMILARITY_MARGIN = 0.03
DEFAULT_SIMILARITY_TOP_K = 5
DEFAULT_CLASSIFIER_MIN_CONFIDENCE = 0.50

# Stage 1 analyses both the original upload and a background-filtered view.
# When filtering succeeds, both views contribute equally to the final gate.
STAGE1_ORIGINAL_WEIGHT = 0.50
STAGE1_FILTERED_WEIGHT = 0.50

DINOV2_CHECKPOINT_NAME = "dinov2_banana_disease.pth"
DINOV2_LABELS_NAME = "dinov2_labels.json"
DINOV2_REFERENCES_NAME = "disease_references.pt"
DINOV2_SIMILARITY_CONFIG_NAME = "similarity_config.json"
DINOV2_INITIAL_REFERENCES_NAME = "initial_references.pt"
DINOV2_INITIAL_CONFIG_NAME = "initial_similarity_config.json"
DINOV2_METRICS_NAME = "dinov2_metrics.json"

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    stage: str
    key: str
    display_name: str
    model_type: str
    model: Any
    processor_or_transform: Any
    classes: list[str]
    folder: Path


_MODEL_LOCK = threading.RLock()
_LOADED_MODELS: dict[str, LoadedModel | None] = {
    "initial": None,
    "health": None,
}
_REFERENCE_BANK_CACHE: dict[Path, dict[str, Any]] = {}
_MODEL_METRICS_CACHE: dict[Path, dict[str, Any] | None] = {}


def trained_ai_root() -> Path:
    return Path(settings.TRAINED_AI_DIR).expanduser().resolve()


def initial_scan_ai_root() -> Path:
    # Kept as a compatibility helper for older imports. Stage 1 now uses the
    # trained DINOv2 model under TRAINED_AI_DIR instead of INITIAL_SCAN_AI_DIR.
    return trained_ai_root()


def _root_for_stage(stage: str) -> Path:
    if stage in {"initial", "health"}:
        return trained_ai_root()
    raise ValueError(f"Unknown inference stage: {stage}")


def model_folder(model_key: str, stage: str) -> Path:
    """Return the DINOv2 folder used by both inference stages."""
    if model_key != "dinov2-base":
        raise PredictionError("Only DINOv2 Base is supported by this project.")
    if stage not in {"initial", "health"}:
        raise ValueError(f"Unknown inference stage: {stage}")
    return trained_ai_root() / MODEL_SPECS[model_key]["folder"]


def best_model_path(model_key: str, stage: str) -> Path:
    """Backward-compatible alias used elsewhere in the project."""
    return model_folder(model_key, stage)


def _custom_dinov2_files_present(folder: Path, stage: str) -> bool:
    common = (
        (folder / DINOV2_CHECKPOINT_NAME).exists()
        and (folder / DINOV2_LABELS_NAME).exists()
    )
    if not common:
        return False

    if stage == "initial":
        return (folder / DINOV2_INITIAL_REFERENCES_NAME).exists()
    if stage == "health":
        return (folder / DINOV2_REFERENCES_NAME).exists()
    return False


def model_is_available(model_key: str, stage: str) -> bool:
    if model_key != "dinov2-base":
        return False

    folder = model_folder(model_key, stage)
    return folder.is_dir() and _custom_dinov2_files_present(folder, stage)


def get_model_choices() -> list[dict[str, Any]]:
    """Return the single Stage-2 DINOv2 model entry used by the UI."""
    initial_available = model_is_available(INITIAL_SCAN_MODEL_KEY, "initial")
    health_available = model_is_available("dinov2-base", "health")
    spec = MODEL_SPECS["dinov2-base"]

    return [
        {
            "key": "dinov2-base",
            "display_name": spec["display_name"],
            "description": spec["description"],
            "available": health_available,
            "initial_available": initial_available,
            "health_available": health_available,
            "missing_text": (
                "" if health_available else "DINOv2 disease model / similarity references"
            ),
            "selected": health_available,
        }
    ]


def _safe_torch_load(path: Path) -> Any:
    """Load one of this project's own PyTorch checkpoint files."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases that do not support the
        # weights_only keyword.
        return torch.load(path, map_location="cpu")


def _labels_from_json(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, dict):
        try:
            return [str(raw[str(index)]) for index in range(len(raw))]
        except (KeyError, ValueError):
            # Fallback for a dict with numeric-looking keys in unusual order.
            ordered = sorted(raw.items(), key=lambda item: int(item[0]))
            return [str(value) for _, value in ordered]

    if isinstance(raw, list):
        return [str(value) for value in raw]

    raise ModelNotAvailableError(f"Could not understand class labels in: {path}")


def _load_custom_dinov2(folder: Path, stage: str) -> LoadedModel:
    """Load the user's trained timm DINOv2 Base checkpoint for either stage."""
    checkpoint_path = folder / DINOV2_CHECKPOINT_NAME
    labels_path = folder / DINOV2_LABELS_NAME

    checkpoint = _safe_torch_load(checkpoint_path)
    classes = [str(value) for value in checkpoint.get("classes", [])]
    if not classes:
        classes = _labels_from_json(labels_path)

    model_name = str(
        checkpoint.get("model_name", "vit_base_patch14_dinov2.lvd142m")
    )
    image_size = int(checkpoint.get("image_size", 518))

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(classes),
    )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ModelNotAvailableError(
            f"The checkpoint at {checkpoint_path} does not contain model_state_dict."
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return LoadedModel(
        stage=stage,
        key="dinov2-base",
        display_name=(
            "DINOv2 Base Banana/Not-Banana gate"
            if stage == "initial"
            else "DINOv2 Base"
        ),
        model_type="timm_dinov2",
        model=model,
        processor_or_transform=eval_transform,
        classes=classes,
        folder=folder,
    )


def _release_stage_model(stage: str) -> None:
    runtime = _LOADED_MODELS.get(stage)
    if runtime is not None:
        try:
            runtime.model.to("cpu")
        except Exception:
            pass

    _LOADED_MODELS[stage] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_or_load_model(model_key: str, stage: str) -> LoadedModel:
    if model_key not in MODEL_SPECS:
        raise PredictionError("Unknown AI model selected.")

    if stage not in _LOADED_MODELS:
        raise PredictionError("Unknown inference stage requested.")

    current = _LOADED_MODELS[stage]
    if current is not None and current.key == model_key:
        return current

    folder = model_folder(model_key, stage)
    if not model_is_available(model_key, stage):
        stage_name = "initial banana-leaf scan" if stage == "initial" else "health scan"
        extra = ""
        if model_key == "dinov2-base":
            reference_name = (
                DINOV2_INITIAL_REFERENCES_NAME
                if stage == "initial"
                else DINOV2_REFERENCES_NAME
            )
            extra = (
                f" Required files: {DINOV2_CHECKPOINT_NAME}, {DINOV2_LABELS_NAME}, "
                f"and {reference_name}."
            )
        raise ModelNotAvailableError(
            f"The {stage_name} files for {MODEL_SPECS[model_key]['display_name']} "
            f"were not found at: {folder}.{extra}"
        )

    _release_stage_model(stage)
    runtime = _load_custom_dinov2(folder, stage)
    _LOADED_MODELS[stage] = runtime
    return runtime


def _open_image(uploaded_file) -> Image.Image:
    try:
        image = Image.open(uploaded_file)
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError(
            "The uploaded file could not be read as an image. "
            "Please use a JPG, JPEG, PNG or WEBP image."
        ) from error


def _run_runtime(image: Image.Image, runtime: LoadedModel) -> torch.Tensor:
    if runtime.model_type != "timm_dinov2":
        raise PredictionError("Only DINOv2 Base is supported by this project.")

    with torch.inference_mode():
        tensor = runtime.processor_or_transform(image).unsqueeze(0).to(DEVICE)
        logits = runtime.model(tensor)
        return torch.softmax(logits.float(), dim=1)[0].cpu()


def _extract_dinov2_embedding(image: Image.Image, runtime: LoadedModel) -> torch.Tensor:
    """Return one L2-normalised feature vector from the fine-tuned DINOv2."""
    if runtime.model_type != "timm_dinov2":
        raise PredictionError("Similarity embeddings are only configured for DINOv2 Base.")

    tensor = runtime.processor_or_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        features = runtime.model.forward_features(tensor)

        # timm VisionTransformer/DINOv2 exposes the representation immediately
        # before the classification layer with forward_head(..., pre_logits=True).
        try:
            embedding = runtime.model.forward_head(features, pre_logits=True)
        except TypeError:
            if features.ndim == 3:
                embedding = features[:, 0]
            elif features.ndim == 4:
                embedding = features.mean(dim=(2, 3))
            else:
                embedding = features.flatten(1)

        embedding = embedding.float().flatten(1)
        embedding = F.normalize(embedding, dim=1)

    return embedding[0].cpu()


def _normalise_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _match_reference_class(reference_classes: list[str], wanted: str) -> str | None:
    wanted_normalised = _normalise_label(wanted)
    for candidate in reference_classes:
        if _normalise_label(candidate) == wanted_normalised:
            return candidate
    return None


def _load_reference_bank(folder: Path, filename: str) -> dict[str, Any]:
    path = folder / filename
    cached = _REFERENCE_BANK_CACHE.get(path)
    if cached is not None:
        return cached

    if not path.exists():
        builder = (
            "build_initial_references.py"
            if filename == DINOV2_INITIAL_REFERENCES_NAME
            else "build_disease_references.py"
        )
        raise ModelNotAvailableError(
            f"Similarity reference file not found: {path}. "
            f"Run {builder} once before starting Django."
        )

    bank = _safe_torch_load(path)
    embeddings = bank.get("embeddings") if isinstance(bank, dict) else None
    if not isinstance(embeddings, dict) or not embeddings:
        raise ModelNotAvailableError(
            f"The similarity reference file is invalid: {path}"
        )

    normalised_embeddings: dict[str, torch.Tensor] = {}
    for class_name, values in embeddings.items():
        tensor = torch.as_tensor(values, dtype=torch.float32).cpu()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            continue
        normalised_embeddings[str(class_name)] = F.normalize(tensor, dim=1)

    if not normalised_embeddings:
        raise ModelNotAvailableError(
            f"No usable embeddings were found in: {path}"
        )

    bank = dict(bank)
    bank["embeddings"] = normalised_embeddings
    _REFERENCE_BANK_CACHE[path] = bank
    return bank


def _similarity_settings(folder: Path) -> dict[str, float | int]:
    settings_values: dict[str, float | int] = {
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "similarity_margin": DEFAULT_SIMILARITY_MARGIN,
        "top_k": DEFAULT_SIMILARITY_TOP_K,
        "classifier_min_confidence": DEFAULT_CLASSIFIER_MIN_CONFIDENCE,
    }

    config_path = folder / DINOV2_SIMILARITY_CONFIG_NAME
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                supplied = json.load(file)

            settings_values["similarity_threshold"] = float(
                supplied.get(
                    "similarity_threshold",
                    settings_values["similarity_threshold"],
                )
            )
            settings_values["similarity_margin"] = float(
                supplied.get("similarity_margin", settings_values["similarity_margin"])
            )
            settings_values["top_k"] = max(
                1,
                int(supplied.get("top_k", settings_values["top_k"])),
            )
            settings_values["classifier_min_confidence"] = float(
                supplied.get(
                    "classifier_min_confidence",
                    settings_values["classifier_min_confidence"],
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("Could not read %s; using default similarity settings", config_path)

    settings_values["similarity_threshold"] = max(
        -1.0, min(1.0, float(settings_values["similarity_threshold"]))
    )
    settings_values["similarity_margin"] = max(
        0.0, min(2.0, float(settings_values["similarity_margin"]))
    )
    settings_values["classifier_min_confidence"] = max(
        0.0, min(1.0, float(settings_values["classifier_min_confidence"]))
    )
    return settings_values


def _compare_to_reference_bank(
    query_embedding: torch.Tensor,
    runtime: LoadedModel,
) -> dict[str, Any]:
    bank = _load_reference_bank(runtime.folder, DINOV2_REFERENCES_NAME)
    config = _similarity_settings(runtime.folder)
    top_k = int(config["top_k"])

    embeddings: dict[str, torch.Tensor] = bank["embeddings"]
    scores: dict[str, float] = {}
    top_examples: dict[str, list[float]] = {}

    query = F.normalize(query_embedding.float().cpu().flatten(), dim=0)

    for class_name, class_embeddings in embeddings.items():
        similarities = class_embeddings @ query
        k = min(top_k, int(similarities.numel()))
        top_values = torch.topk(similarities, k=k).values

        # Nearest examples answer the question "does this look like verified
        # examples of this class?". A small prototype component makes the
        # score less sensitive to one accidental nearest neighbour.
        nearest_score = float(top_values.mean().item())
        prototype = F.normalize(class_embeddings.mean(dim=0), dim=0)
        prototype_score = float(torch.dot(prototype, query).item())
        combined_score = 0.75 * nearest_score + 0.25 * prototype_score

        scores[class_name] = combined_score
        top_examples[class_name] = [float(value.item()) for value in top_values]

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_class, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -1.0
    margin = best_score - second_score

    return {
        "best_class": best_class,
        "best_score": best_score,
        "second_score": second_score,
        "margin": margin,
        "scores": scores,
        "top_examples": top_examples,
        "threshold": float(config["similarity_threshold"]),
        "required_margin": float(config["similarity_margin"]),
        "top_k": top_k,
        "classifier_min_confidence": float(config["classifier_min_confidence"]),
    }


def _initial_similarity_settings(folder: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "banana_threshold": DEFAULT_INITIAL_BANANA_THRESHOLD,
        "binary_margin": DEFAULT_INITIAL_BINARY_MARGIN,
        "top_k": DEFAULT_SIMILARITY_TOP_K,
    }

    config_path = folder / DINOV2_INITIAL_CONFIG_NAME
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                supplied = json.load(file)

            # Stage 1 is intentionally fixed at 65% in this project version.
            # Existing config files may still contain the older auto-generated
            # threshold, so do not let them silently override the new value.
            values["banana_threshold"] = DEFAULT_INITIAL_BANANA_THRESHOLD
            values["binary_margin"] = float(
                supplied.get("binary_margin", values["binary_margin"])
            )
            values["top_k"] = max(1, int(supplied.get("top_k", values["top_k"])))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception(
                "Could not read %s; using default Stage 1 similarity settings",
                config_path,
            )

    values["banana_threshold"] = max(
        -1.0, min(1.0, float(values["banana_threshold"]))
    )
    values["binary_margin"] = max(0.0, min(2.0, float(values["binary_margin"])))
    return values


def _compare_to_initial_reference_bank(
    query_embedding: torch.Tensor,
    runtime: LoadedModel,
) -> dict[str, Any]:
    """Compare an upload only against Banana Leaf and Not BananaLeaf banks."""
    bank = _load_reference_bank(runtime.folder, DINOV2_INITIAL_REFERENCES_NAME)
    config = _initial_similarity_settings(runtime.folder)
    top_k = int(config["top_k"])

    embeddings: dict[str, torch.Tensor] = bank["embeddings"]
    required = {"banana_leaf", "not_bananaleaf"}
    available = {_normalise_label(name): name for name in embeddings}
    if not required.issubset(available):
        raise ModelNotAvailableError(
            f"{DINOV2_INITIAL_REFERENCES_NAME} must contain 'Banana Leaf' and "
            "'Not BananaLeaf' reference groups."
        )

    query = F.normalize(query_embedding.float().cpu().flatten(), dim=0)
    scores: dict[str, float] = {}
    top_examples: dict[str, list[float]] = {}

    for output_name, key in (("Banana Leaf", "banana_leaf"), ("Not BananaLeaf", "not_bananaleaf")):
        class_name = available[key]
        class_embeddings = embeddings[class_name]
        similarities = class_embeddings @ query
        k = min(top_k, int(similarities.numel()))
        top_values = torch.topk(similarities, k=k).values
        nearest_score = float(top_values.mean().item())
        prototype = F.normalize(class_embeddings.mean(dim=0), dim=0)
        prototype_score = float(torch.dot(prototype, query).item())
        combined_score = 0.75 * nearest_score + 0.25 * prototype_score
        scores[output_name] = combined_score
        top_examples[output_name] = [float(value.item()) for value in top_values]

    banana_score = scores["Banana Leaf"]
    not_banana_score = scores["Not BananaLeaf"]
    best_class = "Banana Leaf" if banana_score >= not_banana_score else "Not BananaLeaf"
    best_score = max(banana_score, not_banana_score)
    margin = abs(banana_score - not_banana_score)

    accepted = (
        best_class == "Banana Leaf"
        and banana_score >= float(config["banana_threshold"])
        and (banana_score - not_banana_score) >= float(config["binary_margin"])
    )

    return {
        "best_class": best_class,
        "best_score": best_score,
        "banana_score": banana_score,
        "not_banana_score": not_banana_score,
        "margin": margin,
        "scores": scores,
        "top_examples": top_examples,
        "banana_threshold": float(config["banana_threshold"]),
        "required_margin": float(config["binary_margin"]),
        "top_k": top_k,
        "accepted": accepted,
    }


def _find_class_index(classes: list[str], wanted: str) -> int:
    wanted_normalised = wanted.strip().lower().replace("_", " ")
    for index, label in enumerate(classes):
        label_normalised = str(label).strip().lower().replace("_", " ")
        if label_normalised == wanted_normalised:
            return index
    raise PredictionError(
        f"The model classes {classes} do not contain the expected class '{wanted}'."
    )


def _display_similarity(value: float) -> float:
    """Clamp cosine similarity to 0..1 for progress-bar display only."""
    return max(0.0, min(1.0, float(value)))


def _image_to_data_url(
    image: Image.Image,
    *,
    max_side: int = 900,
    quality: int = 82,
) -> str:
    """Return a compact JPEG data URL for browser display.

    The result page stores scan output in sessionStorage, so keep this preview
    intentionally compact rather than sending a full-resolution image.
    """
    preview = image.convert("RGB").copy()
    preview.thumbnail((max_side, max_side))

    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _slug_health_label(label: str) -> str:
    normalised = label.strip().lower().replace("-", "_").replace(" ", "_")
    if "sigatoka" in normalised:
        return "black_sigatoka"
    if "healthy" in normalised:
        return "healthy"
    return normalised


def _display_health_label(label_slug: str, original_label: str) -> str:
    if label_slug == "black_sigatoka":
        return "Black Sigatoka Detected"
    if label_slug == "healthy":
        return "Healthy"
    if label_slug == "unknown":
        return "Unknown / No Reliable Match"
    return original_label.replace("_", " ").title()


def _probability_dict(classes: list[str], probabilities: torch.Tensor) -> dict[str, float]:
    return {
        str(classes[index]): float(probabilities[index].item())
        for index in range(min(len(classes), len(probabilities)))
    }


def _load_model_metrics(runtime: LoadedModel) -> dict[str, Any] | None:
    """Load saved evaluation metrics for the result page when available.

    DINOv2 metrics are produced by ``evaluate_dinov2_metrics.py`` and stored
    beside the model as ``dinov2_metrics.json``. If that file has not been
    generated yet, the checkpoint's stored test/validation accuracy is still
    exposed so the UI can show the statistics that are genuinely available.
    """
    cache_key = runtime.folder.resolve()
    if cache_key in _MODEL_METRICS_CACHE:
        return _MODEL_METRICS_CACHE[cache_key]

    metrics_path = runtime.folder / (
        DINOV2_METRICS_NAME if runtime.key == "dinov2-base" else "metrics.json"
    )

    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            if isinstance(metrics, dict):
                metrics = dict(metrics)
                metrics["source"] = "evaluation_file"
                _MODEL_METRICS_CACHE[cache_key] = metrics
                return metrics
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("Could not read model metrics from %s", metrics_path)

    if runtime.key == "dinov2-base":
        checkpoint_path = runtime.folder / DINOV2_CHECKPOINT_NAME
        if checkpoint_path.exists():
            try:
                checkpoint = _safe_torch_load(checkpoint_path)
                fallback: dict[str, Any] = {
                    "source": "checkpoint",
                    "note": (
                        "Only accuracy values were stored during training. Run "
                        "evaluate_dinov2_metrics.py once to add precision, recall, "
                        "F1 score, per-class statistics and a confusion matrix."
                    ),
                }
                if "test_accuracy" in checkpoint:
                    value = float(checkpoint["test_accuracy"])
                    fallback["accuracy"] = value / 100.0 if value > 1.0 else value
                if "best_validation_accuracy" in checkpoint:
                    value = float(checkpoint["best_validation_accuracy"])
                    fallback["validation_accuracy"] = (
                        value / 100.0 if value > 1.0 else value
                    )
                if len(fallback) > 2:
                    _MODEL_METRICS_CACHE[cache_key] = fallback
                    return fallback
            except Exception:
                logger.exception("Could not read fallback metrics from %s", checkpoint_path)

    _MODEL_METRICS_CACHE[cache_key] = None
    return None


def predict_two_stage(uploaded_file, model_key: str) -> dict[str, Any]:
    """Run banana-leaf validation, then health analysis and similarity rejection."""

    if model_key not in MODEL_SPECS:
        raise PredictionError("Unknown health AI model selected.")

    image = _open_image(uploaded_file)

    with _MODEL_LOCK:
        # ---------------- Stage 1: binary Banana vs Not Banana gate ----------------
        # The five disease classes are NOT shown or used as Stage 1 options.
        # The upload is embedded by the existing fine-tuned DINOv2 backbone and
        # compared with two reference groups built from the dataset:
        # Banana Leaf and Not BananaLeaf. Only a clear Banana Leaf result moves
        # on to Stage 2.
        initial_runtime = _get_or_load_model(INITIAL_SCAN_MODEL_KEY, "initial")

        # View A: analyse the untouched upload.
        original_embedding = _extract_dinov2_embedding(image, initial_runtime)
        original_similarity = _compare_to_initial_reference_bank(
            original_embedding, initial_runtime
        )

        # View B: isolate the foreground leaf, replace the background with a
        # neutral grey, crop with padding, and analyse that view as well.
        # This is supplementary only: if filtering fails, Stage 1 falls back
        # to the original image instead of rejecting the scan.
        filtered_view = make_background_filtered_view(image)
        filtered_similarity = None
        if filtered_view.applied:
            filtered_embedding = _extract_dinov2_embedding(
                filtered_view.image, initial_runtime
            )
            filtered_similarity = _compare_to_initial_reference_bank(
                filtered_embedding, initial_runtime
            )

        if filtered_similarity is not None:
            original_weight = STAGE1_ORIGINAL_WEIGHT
            filtered_weight = STAGE1_FILTERED_WEIGHT
            weight_total = original_weight + filtered_weight

            banana_similarity = (
                float(original_similarity["banana_score"]) * original_weight
                + float(filtered_similarity["banana_score"]) * filtered_weight
            ) / weight_total
            not_banana_similarity = (
                float(original_similarity["not_banana_score"]) * original_weight
                + float(filtered_similarity["not_banana_score"]) * filtered_weight
            ) / weight_total
        else:
            original_weight = 1.0
            filtered_weight = 0.0
            banana_similarity = float(original_similarity["banana_score"])
            not_banana_similarity = float(original_similarity["not_banana_score"])

        threshold = float(original_similarity["banana_threshold"])
        required_margin = float(original_similarity["required_margin"])
        combined_margin = banana_similarity - not_banana_similarity
        accepted = (
            banana_similarity >= threshold
            and combined_margin >= required_margin
        )
        best_reference_class = (
            "Banana Leaf"
            if banana_similarity >= not_banana_similarity
            else "Not BananaLeaf"
        )

        initial_result = {
            "label": "banana_leaf" if accepted else "not_banana_leaf",
            "label_display": "Banana Leaf" if accepted else "Not Banana Leaf",
            "banana_leaf_similarity": banana_similarity,
            "not_banana_leaf_similarity": not_banana_similarity,
            # Compatibility field for the front end. These are cosine-similarity
            # scores, not calibrated probabilities.
            "banana_leaf_probability": _display_similarity(banana_similarity),
            "confidence": _display_similarity(
                banana_similarity if accepted else not_banana_similarity
            ),
            "best_reference_class": best_reference_class,
            "similarity_scores": {
                "Banana Leaf": banana_similarity,
                "Not BananaLeaf": not_banana_similarity,
            },
            "probabilities": {
                "Banana Leaf": banana_similarity,
                "Not BananaLeaf": not_banana_similarity,
            },
            "threshold": threshold,
            "margin": abs(combined_margin),
            "signed_margin": combined_margin,
            "required_margin": required_margin,
            "dual_view": {
                "enabled": True,
                "background_filter_applied": bool(filtered_view.applied),
                "background_filter_method": filtered_view.method,
                "background_filter_reason": filtered_view.reason,
                "original_weight": original_weight,
                "background_filtered_weight": filtered_weight,
                "background_filtered_preview_url": (
                    _image_to_data_url(filtered_view.image)
                    if filtered_similarity is not None
                    else None
                ),
                "original": {
                    "banana_leaf_similarity": float(original_similarity["banana_score"]),
                    "not_banana_leaf_similarity": float(original_similarity["not_banana_score"]),
                },
                "background_filtered": (
                    {
                        "banana_leaf_similarity": float(filtered_similarity["banana_score"]),
                        "not_banana_leaf_similarity": float(filtered_similarity["not_banana_score"]),
                        "preview_url": _image_to_data_url(filtered_view.image),
                    }
                    if filtered_similarity is not None
                    else None
                ),
            },
            "accepted": accepted,
            "score_type": "cosine_similarity",
            "model": INITIAL_SCAN_MODEL_KEY,
            "model_display": initial_runtime.display_name,
        }

        if not accepted:
            return {
                "accepted": False,
                "initial_scan": initial_result,
                "health_scan": None,
                "model": model_key,
                "model_display": MODEL_SPECS[model_key]["display_name"],
                "initial_model": INITIAL_SCAN_MODEL_KEY,
                "initial_model_display": initial_runtime.display_name,
                "device": DEVICE.type,
            }

        # ---------------- Stage 2: health classification ----------------
        # Stage 1 and Stage 2 use the same fine-tuned DINOv2 checkpoint.
        # Reuse the already-loaded model so the 338 MB checkpoint is not loaded twice.
        _release_stage_model("health")
        health_runtime = initial_runtime

        health_probabilities = _run_runtime(image, health_runtime)

        predicted_index = int(torch.argmax(health_probabilities).item())
        if predicted_index >= len(health_runtime.classes):
            raise PredictionError("The health model returned an unexpected class index.")

        classifier_original = health_runtime.classes[predicted_index]
        classifier_slug = _slug_health_label(classifier_original)
        classifier_confidence = float(health_probabilities[predicted_index].item())

        final_original = classifier_original
        final_slug = classifier_slug
        reliable_match = True
        rejection_reasons: list[str] = []
        similarity_result = None
        decision_checks = None

        # The new DINOv2 health model uses positive-reference similarity.
        if health_runtime.model_type == "timm_dinov2":
            query_embedding = _extract_dinov2_embedding(image, health_runtime)
            similarity_result = _compare_to_reference_bank(query_embedding, health_runtime)

            reference_classes = list(similarity_result["scores"].keys())
            classifier_reference_class = _match_reference_class(
                reference_classes,
                classifier_original,
            )
            similarity_best_class = str(similarity_result["best_class"])

            agreement = (
                classifier_reference_class is not None
                and _normalise_label(classifier_reference_class)
                == _normalise_label(similarity_best_class)
            )
            similarity_pass = (
                float(similarity_result["best_score"])
                >= float(similarity_result["threshold"])
            )
            margin_pass = (
                float(similarity_result["margin"])
                >= float(similarity_result["required_margin"])
            )
            confidence_pass = (
                classifier_confidence
                >= float(similarity_result["classifier_min_confidence"])
            )

            if not agreement:
                rejection_reasons.append(
                    "The classifier and visual-similarity check selected different classes."
                )
            if not similarity_pass:
                rejection_reasons.append(
                    "The image was not similar enough to verified reference examples."
                )
            if not margin_pass:
                rejection_reasons.append(
                    "The best visual match was too close to the next-best class."
                )
            if not confidence_pass:
                rejection_reasons.append(
                    "The classifier confidence was below the minimum acceptance level."
                )

            decision_checks = {
                "classifier_reference_agreement": bool(agreement),
                "similarity_threshold_pass": bool(similarity_pass),
                "similarity_margin_pass": bool(margin_pass),
                "classifier_confidence_pass": bool(confidence_pass),
            }

            reliable_match = agreement and similarity_pass and margin_pass and confidence_pass
            if not reliable_match:
                final_slug = "unknown"
                final_original = "Unknown"

        # Heatmap still explains the classifier's most likely class. If the
        # similarity system rejects the prediction, the UI explicitly says so.
        heatmap_url = None
        heatmap_method = None
        heatmap_error = None
        try:
            heatmap_url, heatmap_method = generate_gradcam_data_url(
                image=image,
                runtime=health_runtime,
                class_index=predicted_index,
            )
        except Exception:
            logger.exception(
                "Grad-CAM generation failed for model %s",
                model_key,
            )
            heatmap_error = (
                "The prediction completed, but an attention heatmap could not "
                "be generated for this scan."
            )

        health_result = {
            "label": final_slug,
            "label_display": _display_health_label(final_slug, final_original),
            "reliable_match": reliable_match,
            "rejection_reasons": rejection_reasons,
            "confidence": classifier_confidence,
            "classifier_label": classifier_slug,
            "classifier_label_display": _display_health_label(
                classifier_slug,
                classifier_original,
            ),
            "classifier_confidence": classifier_confidence,
            "probabilities": _probability_dict(
                health_runtime.classes,
                health_probabilities,
            ),
            "model": model_key,
            "model_display": health_runtime.display_name,
            "heatmap_url": heatmap_url,
            "heatmap_method": heatmap_method,
            "heatmap_target": classifier_original,
            "heatmap_error": heatmap_error,
            "similarity": similarity_result,
            "decision_checks": decision_checks,
            "model_metrics": _load_model_metrics(health_runtime),
        }

        return {
            "accepted": True,
            "initial_scan": initial_result,
            "health_scan": health_result,
            "model": model_key,
            "model_display": health_runtime.display_name,
            "initial_model": INITIAL_SCAN_MODEL_KEY,
            "initial_model_display": initial_runtime.display_name,
            "device": DEVICE.type,
        }
