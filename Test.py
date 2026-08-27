from huggingface_hub import snapshot_download
from pathlib import Path

BASE_FOLDER = Path("Pretrainedai")

models = {
    "dinov2-base": "facebook/dinov2-base",
    "convnextv2-base": "facebook/convnextv2-base-22k-224",
    "swin-base": "microsoft/swin-base-patch4-window7-224",
    "efficientnetv2-s": "timm/efficientnetv2_rw_s.ra2_in1k",
    "vit-base": "google/vit-base-patch16-224",
}

BASE_FOLDER.mkdir(exist_ok=True)

for folder_name, model_name in models.items():

    destination = BASE_FOLDER / folder_name

    print(f"\nDownloading {model_name}")
    print(f"Saving to: {destination}")

    snapshot_download(
        repo_id=model_name,
        local_dir=destination
    )

    print(f"Finished: {folder_name}")

print("\nAll models downloaded.")