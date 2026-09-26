"""Build positive-reference embeddings for the trained DINOv2 disease model.

Run once after placing these files in:
    trainedai/dinov2-base/
        dinov2_banana_disease.pth
        dinov2_labels.json

Example from the repository root:
    python Django/build_disease_references.py --dataset "C:\\Users\\jimmy\\Downloads\\Training\\Datasets\\Banana Leaf"

The script creates:
    trainedai/dinov2-base/disease_references.pt
    trainedai/dinov2-base/similarity_config.json

Django then uses those positive examples to ask whether an uploaded image really
looks sufficiently similar to a known class instead of always forcing one of the
five disease labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CHECKPOINT_NAME = "dinov2_banana_disease.pth"
LABELS_NAME = "dinov2_labels.json"
REFERENCES_NAME = "disease_references.pt"
CONFIG_NAME = "similarity_config.json"


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_labels(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, dict):
        return [str(raw[str(index)]) for index in range(len(raw))]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    raise RuntimeError(f"Could not understand labels in {path}")


def normalise_label(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def find_class_folder(dataset_root: Path, class_name: str) -> Path:
    exact = dataset_root / class_name
    if exact.is_dir():
        return exact

    wanted = normalise_label(class_name)
    for child in dataset_root.iterdir():
        if child.is_dir() and normalise_label(child.name) == wanted:
            return child

    raise FileNotFoundError(
        f"Could not find a folder for class '{class_name}' inside {dataset_root}"
    )


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_batch_embeddings(model, batch: torch.Tensor) -> torch.Tensor:
    features = model.forward_features(batch)
    try:
        embeddings = model.forward_head(features, pre_logits=True)
    except TypeError:
        if features.ndim == 3:
            embeddings = features[:, 0]
        elif features.ndim == 4:
            embeddings = features.mean(dim=(2, 3))
        else:
            embeddings = features.flatten(1)

    embeddings = embeddings.float().flatten(1)
    return F.normalize(embeddings, dim=1)


def build_embeddings_for_class(
    paths: list[Path],
    transform,
    model,
    device: torch.device,
    batch_size: int,
    class_name: str,
) -> tuple[torch.Tensor, list[str]]:
    vectors: list[torch.Tensor] = []
    used_paths: list[str] = []
    pending_tensors: list[torch.Tensor] = []
    pending_paths: list[str] = []

    def flush() -> None:
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device)
        with torch.inference_mode():
            batch_embeddings = extract_batch_embeddings(model, batch).cpu()
        vectors.append(batch_embeddings)
        used_paths.extend(pending_paths)
        pending_tensors.clear()
        pending_paths.clear()

    for index, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                pending_tensors.append(transform(image))
                pending_paths.append(str(path))
        except (UnidentifiedImageError, OSError, ValueError):
            print(f"  Skipping unreadable image: {path}")
            continue

        if len(pending_tensors) >= batch_size:
            flush()

        if index % 50 == 0 or index == len(paths):
            print(f"  {class_name}: {index}/{len(paths)} images")

    flush()

    if not vectors:
        raise RuntimeError(f"No readable images were found for class {class_name}")

    return torch.cat(vectors, dim=0), used_paths


def class_positive_scores(bank: torch.Tensor, top_k: int) -> torch.Tensor:
    """Leave-one-out positive similarity scores for threshold estimation."""
    if bank.shape[0] < 2:
        prototype = F.normalize(bank.mean(dim=0), dim=0)
        return bank @ prototype

    matrix = bank @ bank.T
    matrix.fill_diagonal_(-2.0)
    k = min(top_k, max(1, bank.shape[0] - 1))
    nearest = torch.topk(matrix, k=k, dim=1).values.mean(dim=1)
    prototype = F.normalize(bank.mean(dim=0), dim=0)
    prototype_scores = bank @ prototype
    return 0.75 * nearest + 0.25 * prototype_scores


def score_against_bank(queries: torch.Tensor, bank: torch.Tensor, top_k: int) -> torch.Tensor:
    matrix = queries @ bank.T
    k = min(top_k, bank.shape[0])
    nearest = torch.topk(matrix, k=k, dim=1).values.mean(dim=1)
    prototype = F.normalize(bank.mean(dim=0), dim=0)
    prototype_scores = queries @ prototype
    return 0.75 * nearest + 0.25 * prototype_scores


def suggest_thresholds(embeddings: dict[str, torch.Tensor], top_k: int) -> tuple[float, float]:
    positive_scores = []
    margins = []

    class_names = list(embeddings)
    for class_name in class_names:
        bank = embeddings[class_name]
        own_scores = class_positive_scores(bank, top_k)
        positive_scores.append(own_scores)

        other_score_columns = []
        for other_name in class_names:
            if other_name == class_name:
                continue
            other_score_columns.append(
                score_against_bank(bank, embeddings[other_name], top_k)
            )

        if other_score_columns:
            best_other = torch.stack(other_score_columns, dim=1).max(dim=1).values
            margins.append(own_scores - best_other)

    all_positive = torch.cat(positive_scores)
    p05 = float(torch.quantile(all_positive, 0.05).item())

    # Allow a small amount below the lower tail of the known positive examples.
    similarity_threshold = max(0.50, min(0.95, p05 - 0.02))

    if margins:
        all_margins = torch.cat(margins)
        margin_p05 = float(torch.quantile(all_margins, 0.05).item())
        similarity_margin = max(0.02, min(0.15, margin_p05 - 0.01))
    else:
        similarity_margin = 0.03

    return similarity_threshold, similarity_margin


def main() -> None:
    default_dataset = Path.home() / "Downloads" / "Training" / "Datasets" / "Banana Leaf"
    repo_root = Path(__file__).resolve().parent.parent
    default_model_dir = repo_root / "trainedai" / "dinov2-base"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Folder containing Black_Sigatoka, Cordana, Freckle, Healthy and Pestalotiopsis.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Folder containing the trained DINOv2 .pth and labels JSON.",
    )
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    checkpoint_path = model_dir / CHECKPOINT_NAME
    labels_path = model_dir / LABELS_NAME

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    classes = [str(value) for value in checkpoint.get("classes", [])]
    if not classes:
        classes = load_labels(labels_path)

    model_name = str(checkpoint.get("model_name", "vit_base_patch14_dinov2.lvd142m"))
    image_size = int(checkpoint.get("image_size", 518))
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain model_state_dict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("BUILDING DINOv2 POSITIVE REFERENCE EMBEDDINGS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_root}")
    print(f"Model: {checkpoint_path}")
    print(f"Classes: {classes}")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    rng = random.Random(args.seed)
    embedding_bank: dict[str, torch.Tensor] = {}
    source_paths: dict[str, list[str]] = {}
    source_counts: dict[str, int] = {}

    for class_name in classes:
        class_folder = find_class_folder(dataset_root, class_name)
        all_paths = collect_images(class_folder)
        if not all_paths:
            raise RuntimeError(f"No images found in {class_folder}")

        selected = list(all_paths)
        rng.shuffle(selected)
        if args.max_per_class > 0:
            selected = selected[: args.max_per_class]

        print(f"\n{class_name}: using {len(selected)} of {len(all_paths)} images")
        vectors, used = build_embeddings_for_class(
            selected,
            transform,
            model,
            device,
            max(1, args.batch_size),
            class_name,
        )

        embedding_bank[class_name] = F.normalize(vectors.float(), dim=1).cpu()
        source_paths[class_name] = used
        source_counts[class_name] = len(used)

    references_path = model_dir / REFERENCES_NAME
    torch.save(
        {
            "version": 1,
            "model_name": model_name,
            "image_size": image_size,
            "classes": classes,
            "embeddings": embedding_bank,
            "source_paths": source_paths,
            "source_counts": source_counts,
            "max_per_class": args.max_per_class,
            "seed": args.seed,
        },
        references_path,
    )

    similarity_threshold, similarity_margin = suggest_thresholds(
        embedding_bank,
        max(1, args.top_k),
    )

    config = {
        "similarity_threshold": round(similarity_threshold, 4),
        "similarity_margin": round(similarity_margin, 4),
        "top_k": max(1, args.top_k),
        "classifier_min_confidence": 0.50,
        "note": (
            "Auto-generated starting thresholds. Test with real banana leaves and hard "
            "negative images such as coconut/palm leaves and adjust if necessary."
        ),
    }

    config_path = model_dir / CONFIG_NAME
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Reference embeddings saved to:\n{references_path}")
    print(f"Similarity settings saved to:\n{config_path}")
    print(f"Suggested disease similarity threshold: {similarity_threshold:.4f}")
    print(f"Suggested class-margin threshold: {similarity_margin:.4f}")
    print("\nYou can now start Django normally.")


if __name__ == "__main__":
    main()
