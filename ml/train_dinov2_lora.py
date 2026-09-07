# ml/train_dinov2_lora.py
# Source: https://huggingface.co/docs/transformers/model_doc/dinov2
# https://huggingface.co/facebook/dinov2-base
# https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora

from transformers import AutoImageProcessor, Dinov2ForImageClassification
from peft import LoraConfig, get_peft_model
import torch
from common import get_loaders, train_model

CHECKPOINT = "facebook/dinov2-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
image_size = processor.crop_size["height"]

train_loader, val_loader, classes = get_loaders(
    "../data/processed", image_size, processor.image_mean, processor.image_std
)
num_classes = len(classes)

base_model = Dinov2ForImageClassification.from_pretrained(CHECKPOINT, num_labels=num_classes)
# LoRA trains only ~0.7% of the ~86M parameters -- everything else stays frozen
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1)
model = get_peft_model(base_model, lora_config)

if __name__ == "__main__":
    model.print_trainable_parameters()
    train_model(
        model, train_loader, val_loader, device,
        epochs=15, lr=1e-3, save_path="models/dinov2_lora_raw.pt",
    )
    model.save_pretrained("models/dinov2_lora_adapter")  # saves just the small adapter
