"""Cache the pretrained HF backbones inside the IMAGE, not at first request.

The website loads local trained .pt state dictionaries separately from these
base architectures. This runs at Docker build time, while network is available.
"""
import os
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

MODEL = os.environ.get("ACTIVE_MODEL", "convnextv2")
HF_IDS = {
    "convnextv2": "facebook/convnextv2-base-22k-224",
    "swin": "microsoft/swin-base-patch4-window7-224",
    "vit": "google/vit-base-patch16-224",
}

print("Caching DINOv2 gate backbone...")
AutoImageProcessor.from_pretrained("facebook/dinov2-base")
AutoModel.from_pretrained("facebook/dinov2-base")
if MODEL in HF_IDS:
    path = HF_IDS[MODEL]
    print("Caching classifier backbone:", path)
    AutoImageProcessor.from_pretrained(path)
    AutoModelForImageClassification.from_pretrained(
        path, num_labels=2, ignore_mismatched_sizes=True
    )
else:
    print("Selected classifier is timm-based and loads its trained local state dict directly.")
print("Backbone cache populated.")
