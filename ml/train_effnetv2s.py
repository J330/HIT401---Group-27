# ml/train_effnetv2s.py
# Source: https://huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k
# https://huggingface.co/docs/timm/models/efficientnet

import timm
import torch
from common import get_loaders, train_model

CHECKPOINT = "efficientnetv2_rw_s"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read the checkpoint's expected input size / normalisation stats
temp = timm.create_model(CHECKPOINT, pretrained=True)
cfg = temp.pretrained_cfg

train_loader, val_loader, classes = get_loaders(
    "../data/processed", image_size=cfg["input_size"][-1], mean=cfg["mean"], std=cfg["std"]
)
num_classes = len(classes)  # 2: healthy, black_sigatoka

model = timm.create_model(CHECKPOINT, pretrained=True, num_classes=num_classes)
# Full fine-tune -- this is the lightest of the five models, no freezing needed.

if __name__ == "__main__":
    train_model(
        model, train_loader, val_loader, device,
        epochs=15, lr=3e-4, save_path="models/effnetv2s.pt", is_hf=False,
    )
