# ml/train_swin.py
# Source: https://huggingface.co/docs/transformers/model_doc/swin
#         https://huggingface.co/microsoft/swin-base-patch4-window7-224
#         Swin Transformer paper -- Liu et al. (2021), https://arxiv.org/abs/2103.14030
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from common import get_loaders, train_model

CHECKPOINT = "microsoft/swin-base-patch4-window7-224"
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
# Swin is hierarchical -- its last block is encoder.layers.3, not stages.3
for name, param in model.named_parameters():
    if "encoder.layers.3" not in name and "classifier" not in name:
        param.requires_grad = False

if __name__ == "__main__":
    train_model(
        model, train_loader, val_loader, device,
        epochs=15, lr=1e-4, save_path="models/swinbase.pt",
    )
