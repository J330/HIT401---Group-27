# ml/common.py
# Sources: Albumentations docs -- https://albumentations.ai/docs/
#torchvision ImageFolder -- https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size, mean, std, train=True):
    # train=True adds random flips/rotation/brightness so the model doesn't
    # just memorise the exact training photos.
    aug = [A.Resize(image_size, image_size)]
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=20, p=0.5),
        ]
    aug += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(aug)


class AlbuImageFolder(ImageFolder):
    # Same as torchvision's ImageFolder (reads class-per-folder images), but
    # runs an Albumentations pipeline instead of torchvision transforms.
    def __init__(self, root, alb_transform):
        super().__init__(root, transform=None)
        self.alb_transform = alb_transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = np.array(Image.open(path).convert("RGB"))
        image = self.alb_transform(image=image)["image"]
        return image, label


def get_loaders(data_dir, image_size, mean, std, batch_size=32):
    train_tf = build_transforms(image_size, mean, std, train=True)
    val_tf = build_transforms(image_size, mean, std, train=False)
    train_ds = AlbuImageFolder(f"{data_dir}/train", train_tf)
    val_ds = AlbuImageFolder(f"{data_dir}/val", val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes


def train_model(model, train_loader, val_loader, device, epochs, lr, save_path, is_hf=True):
    # is_hf=True for Hugging Face models (ConvNeXt V2, Swin, ViT, DINOv2).
    # is_hf=False for timm models (EfficientNetV2-S).
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images) if is_hf else model(images)
            logits = outputs.logits if is_hf else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images) if is_hf else model(images)
                logits = outputs.logits if is_hf else outputs
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(
            f"Epoch {epoch+1}/{epochs} | train_loss={running_loss/len(train_loader.dataset):.4f}"
            f" | val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  saved new best ({val_acc:.4f}) -> {save_path}")

    print(f"Training complete. Best val accuracy: {best_acc:.4f}")
    return model
