# ml/evaluate.py
# Loads each trained checkpoint and reports accuracy + a confusion matrix,
# so you can see which of the four closed-set candidates to deploy.
# Source: scikit-learn docs -- classification_report / confusion_matrix
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
import torch
from sklearn.metrics import classification_report, confusion_matrix
from common import get_loaders
import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_eval(model, val_loader, classes, is_hf):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(pixel_values=images) if is_hf else model(images)
            logits = outputs.logits if is_hf else outputs
            all_preds += logits.argmax(dim=1).cpu().tolist()
            all_labels += labels.tolist()
    print(classification_report(all_labels, all_preds, target_names=classes))
    print(confusion_matrix(all_labels, all_preds))
    return sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)


def evaluate_timm(checkpoint, weights_path):
    temp = timm.create_model(checkpoint, pretrained=True)
    cfg = temp.pretrained_cfg
    _, val_loader, classes = get_loaders(
        "../data/processed", cfg["input_size"][-1], cfg["mean"], cfg["std"]
    )
    model = timm.create_model(checkpoint, pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return run_eval(model, val_loader, classes, is_hf=False)


def evaluate_hf(checkpoint, weights_path):
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    image_size = processor.size.get("shortest_edge", processor.size.get("height"))
    _, val_loader, classes = get_loaders(
        "../data/processed", image_size, processor.image_mean, processor.image_std
    )
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint, num_labels=len(classes), ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return run_eval(model, val_loader, classes, is_hf=True)


if __name__ == "__main__":
    # Keys match inference.py's MODEL_REGISTRY exactly -- whichever wins here
    # is the string you set ACTIVE_MODEL to (see backend/backend/settings.py).
    results = {
        "efficientnetv2s": evaluate_timm("efficientnetv2_rw_s", "models/effnetv2s.pt"),
        "convnextv2": evaluate_hf("facebook/convnextv2-base-22k-224", "models/convnextv2base.pt"),
        "swin": evaluate_hf("microsoft/swin-base-patch4-window7-224", "models/swinbase.pt"),
        "vit": evaluate_hf("google/vit-base-patch16-224", "models/vitbase.pt"),
    }

    print("\n=== Summary ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {acc:.4f}")
    winner = max(results, key=results.get)
    print(f"\nWinner: {winner}")
    print(f"Set ACTIVE_MODEL={winner} in backend settings to make it serve the website.")
