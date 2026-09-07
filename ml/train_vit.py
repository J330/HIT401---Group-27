# ml/train_vit.py
# Source: https://huggingface.co/docs/transformers/model_doc/vit
# https://huggingface.co/google/vit-base-patch16-224

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from common import get_loaders, train_model

CHECKPOINT = "google/vit-base-patch16-224"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
image_size = processor.size.get("shortest_edge", processor.size.get("height"))

train_loader, val_loader, classes = get_loaders(
    "../data/processed", image_size, processor.image_mean, processor.image_std
)
num_classes = len(classes)

model = AutoModelForImageClassification.from_pretrained(
    CHECKPOINT, num_labels=num_classes, ignore_mismatched_sizes=True
)
# ViT is a plain, non-hierarchical transformer -- unfreeze just the last two layers
for name, param in model.named_parameters():
    if not any(k in name for k in ["encoder.layer.10", "encoder.layer.11", "classifier"]):
        param.requires_grad = False

if __name__ == "__main__":
    train_model(
        model, train_loader, val_loader, device,
        epochs=15, lr=1e-4, save_path="models/vitbase.pt",
    )
