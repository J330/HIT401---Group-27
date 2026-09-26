r"""Evaluate the trained DINOv2 banana-health classifier and save report metrics.

This does NOT retrain the model. It recreates the project's original
80/10/10 stratified split with seed 42, evaluates the held-out 10% test set,
and writes:

    trainedai/dinov2-base/dinov2_metrics.json

The Django result card reads that file and displays Accuracy, Precision,
Recall and F1 score, plus per-class statistics when available.

Example from the repository root:

    python Django\evaluate_dinov2_metrics.py --dataset "C:\\Users\\jimmy\\Downloads\\Training\\Datasets\\Banana Leaf"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CHECKPOINT_NAME = "dinov2_banana_disease.pth"
METRICS_NAME = "dinov2_metrics.json"
RANDOM_SEED = 42


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ImageSubset(Dataset):
    def __init__(self, samples, indices, transform):
        self.samples = samples
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        path, target = self.samples[self.indices[item]]
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                return self.transform(image), int(target)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(f"Could not read test image: {path}") from error


def main() -> None:
    default_dataset = Path.home() / "Downloads" / "Training" / "Datasets" / "Banana Leaf"
    repo_root = Path(__file__).resolve().parent.parent
    default_model_dir = repo_root / "trainedai" / "dinov2-base"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = model_dir / CHECKPOINT_NAME

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    metadata = datasets.ImageFolder(dataset_path)
    classes = metadata.classes
    targets = np.asarray(metadata.targets, dtype=np.int64)
    indices = np.arange(len(metadata))

    if len(classes) < 2:
        raise RuntimeError("At least two classes are required for evaluation.")

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=targets,
    )
    temp_targets = targets[temp_indices]
    _, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_targets,
    )

    checkpoint = safe_torch_load(checkpoint_path)
    checkpoint_classes = [str(value) for value in checkpoint.get("classes", [])]
    if checkpoint_classes and checkpoint_classes != classes:
        raise RuntimeError(
            "Dataset class order does not match the trained model.\n"
            f"Model:   {checkpoint_classes}\nDataset: {classes}"
        )

    model_name = str(checkpoint.get("model_name", "vit_base_patch14_dinov2.lvd142m"))
    image_size = int(checkpoint.get("image_size", 518))
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain model_state_dict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("DINOv2 TEST-SET EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Test images: {len(test_indices)}")

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

    dataset = ImageSubset(metadata.samples, test_indices, transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=torch.cuda.is_available())
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"Batch {batch_index}/{len(loader)}")

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    ).tolist()

    per_class = {}
    for name in classes:
        row = report.get(name, {})
        per_class[name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }

    metrics = {
        "model": "DINOv2 Base",
        "model_name": model_name,
        "dataset": str(dataset_path),
        "split": "10% held-out stratified test set",
        "seed": RANDOM_SEED,
        "test_size": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "note": (
            "Metrics are measured on the recreated seed-42 stratified test split. "
            "If the dataset has changed since training, rerun evaluation and interpret "
            "the values as performance on the current dataset split."
        ),
    }

    output_path = model_dir / METRICS_NAME
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy:          {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Macro Precision:   {macro_precision * 100:.2f}%")
    print(f"Macro Recall:    {macro_recall * 100:.2f}%")
    print(f"Macro F1:        {macro_f1 * 100:.2f}%")
    print(f"Weighted F1:     {weighted_f1 * 100:.2f}%")
    print(f"\nSaved to:\n{output_path}")


if __name__ == "__main__":
    main()
