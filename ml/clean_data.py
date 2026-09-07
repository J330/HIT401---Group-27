# ml/clean_data.py
# Removes any image file that's corrupt or not readable before training.
# Source: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify

import os
from PIL import Image

RAW_DIR = "../data/raw"
CLASSES = ["healthy", "black_sigatoka", "other_disease", "non_banana"]


def clean_folder(folder):
    removed = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as img:
                img.verify()  # raises an error if the file is broken
        except Exception:
            os.remove(path)
            removed += 1
    print(f"{folder}: removed {removed} corrupt/unreadable files")


if __name__ == "__main__":
    for class_name in CLASSES:
        folder = os.path.join(RAW_DIR, class_name)
        if os.path.isdir(folder):
            clean_folder(folder)
