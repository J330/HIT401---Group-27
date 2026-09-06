# ml/split_data.py
# Leakage-safe train/val split. Only healthy and black_sigatoka are split --
# other_disease and non_banana stay untouched, used later only to test the
# open-set gate (build_gate.py / inference.py), never to train a classifier.
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "../data/raw"
OUT_DIR = "../data/processed"
CLASSES = ["healthy", "black_sigatoka"]
VAL_SIZE = 0.2
SEED = 42


def split_class(class_name):
    src = os.path.join(RAW_DIR, class_name)
    files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    train_files, val_files = train_test_split(files, test_size=VAL_SIZE, random_state=SEED)

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        dest = os.path.join(OUT_DIR, split_name, class_name)
        os.makedirs(dest, exist_ok=True)
        for f in split_files:
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
    print(f"{class_name}: {len(train_files)} train, {len(val_files)} val")


if __name__ == "__main__":
    for class_name in CLASSES:
        split_class(class_name)
