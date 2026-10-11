"""Verify trained model files were supplied and are not Git LFS text pointers."""
import argparse
from pathlib import Path

MODELS = {
    "convnextv2": "convnextv2base.pt",
    "efficientnetv2s": "effnetv2s.pt",
    "swin": "swinbase.pt",
    "vit": "vitbase.pt",
}


def validate(model: str, root: Path) -> None:
    if model not in MODELS:
        raise SystemExit(f"Unknown ACTIVE_MODEL {model!r}; choices: {', '.join(MODELS)}")
    for name in ("gate_centroids.pt", MODELS[model]):
        path = root / "ml" / "models" / name
        if not path.is_file():
            raise SystemExit(f"Missing trained checkpoint: {path}. Copy it from your trained ml/models folder or download it from Cloud Storage in cloudbuild.yaml.")
        with path.open("rb") as stream:
            start = stream.read(128)
        if start.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(f"{path} is a Git LFS pointer, NOT real model weights. Retrieve actual .pt bytes (e.g. via Cloud Storage).")
        if path.stat().st_size < 1024:
            raise SystemExit(f"{path} looks too small to be a model checkpoint ({path.stat().st_size} bytes).")
        print(f"Model ready: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnextv2", choices=MODELS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    validate(arguments.model, arguments.root)
