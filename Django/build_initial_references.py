r"""Build the binary Stage-1 DINOv2 reference bank.

This script does NOT retrain DINOv2. It loads the already-trained disease model
and uses its feature extractor to build two visual reference groups only:

    Banana Leaf
    Not BananaLeaf

Expected dataset layout by default:
    C:\Users\<user>\Downloads\Training\Datasets\
        Banana Leaf\
            Black_Sigatoka\
            Cordana\
            Freckle\
            Healthy\
            Pestalotiopsis\
        Not BananaLeaf\
            ...images or subfolders...

Run from the repository root:
    python Django\build_initial_references.py

Or provide the dataset root explicitly:
    python Django\build_initial_references.py --dataset-root "C:\\Users\\jimmy\\Downloads\\Training\\Datasets"

The script creates:
    trainedai/dinov2-base/initial_references.pt
    trainedai/dinov2-base/initial_similarity_config.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CHECKPOINT_NAME = "dinov2_banana_disease.pth"
LABELS_NAME = "dinov2_labels.json"
REFERENCES_NAME = "initial_references.pt"
CONFIG_NAME = "initial_similarity_config.json"


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


def build_embeddings(
    paths: list[Path],
    transform,
    model,
    device: torch.device,
    batch_size: int,
    group_name: str,
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
            print(f"  {group_name}: {index}/{len(paths)} images")

    flush()

    if not vectors:
        raise RuntimeError(f"No readable images were found for {group_name}")

    return F.normalize(torch.cat(vectors, dim=0).float(), dim=1), used_paths


def leave_one_out_positive_scores(bank: torch.Tensor, top_k: int) -> torch.Tensor:
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


def score_queries_against_bank(
    queries: torch.Tensor,
    bank: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    matrix = queries @ bank.T
    k = min(top_k, bank.shape[0])
    nearest = torch.topk(matrix, k=k, dim=1).values.mean(dim=1)
    prototype = F.normalize(bank.mean(dim=0), dim=0)
    prototype_scores = queries @ prototype
    return 0.75 * nearest + 0.25 * prototype_scores


def sample_banana_images(
    banana_root: Path,
    rng: random.Random,
    max_per_class: int,
) -> tuple[list[Path], dict[str, int]]:
    """Sample each disease/health subfolder separately so Healthy cannot dominate."""
    class_folders = sorted(path for path in banana_root.iterdir() if path.is_dir())

    # If there are no class subfolders, fall back to all images recursively.
    if not class_folders:
        all_paths = collect_images(banana_root)
        rng.shuffle(all_paths)
        if max_per_class > 0:
            all_paths = all_paths[:max_per_class]
        return all_paths, {"Banana Leaf": len(all_paths)}

    selected: list[Path] = []
    counts: dict[str, int] = {}

    for folder in class_folders:
        paths = collect_images(folder)
        rng.shuffle(paths)
        if max_per_class > 0:
            paths = paths[:max_per_class]
        selected.extend(paths)
        counts[folder.name] = len(paths)

    rng.shuffle(selected)
    return selected, counts


def main() -> None:
    default_dataset_root = Path.home() / "Downloads" / "Training" / "Datasets"
    repo_root = Path(__file__).resolve().parent.parent
    default_model_dir = repo_root / "trainedai" / "dinov2-base"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help="Folder containing both 'Banana Leaf' and 'Not BananaLeaf'.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Folder containing the trained DINOv2 .pth and labels JSON.",
    )
    parser.add_argument(
        "--max-per-banana-class",
        type=int,
        default=250,
        help="Maximum references sampled from each banana disease/health subfolder.",
    )
    parser.add_argument(
        "--max-not-banana",
        type=int,
        default=1250,
        help="Maximum Not BananaLeaf references.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    banana_root = dataset_root / "Banana Leaf"
    not_banana_root = dataset_root / "Not BananaLeaf"
    model_dir = args.model_dir.expanduser().resolve()

    checkpoint_path = model_dir / CHECKPOINT_NAME
    labels_path = model_dir / LABELS_NAME

    if not banana_root.is_dir():
        raise FileNotFoundError(f"Banana Leaf folder not found: {banana_root}")
    if not not_banana_root.is_dir():
        raise FileNotFoundError(f"Not BananaLeaf folder not found: {not_banana_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    classes = [str(value) for value in checkpoint.get("classes", [])]
    if not classes:
        classes = load_labels(labels_path)

    model_name = str(
        checkpoint.get("model_name", "vit_base_patch14_dinov2.lvd142m")
    )
    image_size = int(checkpoint.get("image_size", 518))
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain model_state_dict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("BUILDING DINOv2 STAGE-1 BANANA / NOT-BANANA REFERENCES")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Model: {checkpoint_path}")

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

    banana_paths, banana_source_counts = sample_banana_images(
        banana_root,
        rng,
        max(0, args.max_per_banana_class),
    )
    not_banana_paths = collect_images(not_banana_root)
    rng.shuffle(not_banana_paths)
    if args.max_not_banana > 0:
        not_banana_paths = not_banana_paths[: args.max_not_banana]

    if not banana_paths:
        raise RuntimeError(f"No banana images found under {banana_root}")
    if not not_banana_paths:
        raise RuntimeError(f"No non-banana images found under {not_banana_root}")

    print("\nBanana Leaf reference sampling by source folder:")
    for name, count in banana_source_counts.items():
        print(f"  {name}: {count}")
    print(f"  TOTAL: {len(banana_paths)}")
    print(f"\nNot BananaLeaf references: {len(not_banana_paths)}")

    print("\nExtracting Banana Leaf embeddings...")
    banana_embeddings, banana_used = build_embeddings(
        banana_paths,
        transform,
        model,
        device,
        max(1, args.batch_size),
        "Banana Leaf",
    )

    print("\nExtracting Not BananaLeaf embeddings...")
    not_banana_embeddings, not_banana_used = build_embeddings(
        not_banana_paths,
        transform,
        model,
        device,
        max(1, args.batch_size),
        "Not BananaLeaf",
    )

    embedding_bank = {
        "Banana Leaf": banana_embeddings.cpu(),
        "Not BananaLeaf": not_banana_embeddings.cpu(),
    }

    references_path = model_dir / REFERENCES_NAME
    torch.save(
        {
            "version": 1,
            "model_name": model_name,
            "image_size": image_size,
            "classes": ["Banana Leaf", "Not BananaLeaf"],
            "embeddings": embedding_bank,
            "source_paths": {
                "Banana Leaf": banana_used,
                "Not BananaLeaf": not_banana_used,
            },
            "source_counts": {
                "Banana Leaf": len(banana_used),
                "Not BananaLeaf": len(not_banana_used),
            },
            "banana_source_counts": banana_source_counts,
            "seed": args.seed,
        },
        references_path,
    )

    top_k = max(1, args.top_k)
    banana_own = leave_one_out_positive_scores(banana_embeddings, top_k)
    banana_vs_not = score_queries_against_bank(
        banana_embeddings,
        not_banana_embeddings,
        top_k,
    )

    # Stage 1 uses a fixed 65% Banana Leaf similarity threshold. The binary
    # margin is still estimated from the reference data so Banana Leaf should
    # also clearly beat the Not BananaLeaf reference bank.
    banana_threshold = 0.65

    banana_margins = banana_own - banana_vs_not
    margin_p05 = float(torch.quantile(banana_margins, 0.05).item())
    binary_margin = max(0.01, min(0.15, margin_p05 - 0.005))

    config = {
        "banana_threshold": round(banana_threshold, 4),
        "binary_margin": round(binary_margin, 4),
        "top_k": top_k,
        "note": (
            "Stage 1 Banana Leaf threshold fixed at 0.65. The binary margin is "
            "estimated from the current reference bank."
        ),
    }

    config_path = model_dir / CONFIG_NAME
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"Binary reference bank saved to:\n{references_path}")
    print(f"Stage 1 settings saved to:\n{config_path}")
    print(f"Banana similarity threshold: {banana_threshold:.4f}")
    print(f"Required Banana-vs-Not-Banana margin: {binary_margin:.4f}")
    print("\nStage 1 will now show only: Banana Leaf / Not Banana Leaf.")


if __name__ == "__main__":
    main()
