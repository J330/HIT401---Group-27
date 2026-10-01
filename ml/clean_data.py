# ml/clean_data.py


# References

# https://www.geeksforgeeks.org/image-blur-detection-using-opencv/

# OpenCV tutorial: https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html

# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify

# https://en.wikipedia.org/wiki/RGBA_color_model

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = "../data/raw"
OUT_DIR = "../data/cleaned"
PROCESSED_DIR = "../data/processed"
CLASSES = ["healthy", "black_sigatoka"]

# Tunable settings
BLACK_CORNER = 40          # pixel brightness < this counts as "black"
CORNER_FRAC = 0.12         # corner sample size as fraction of image
MIN_BLACK_CORNERS = 2      # how many black corners = studio shot
BORDER_PX = 10             # border thickness for edge-black check
BORDER_RATIO = 0.40        # fraction of border that must be black
BLUR_THRESHOLD = 12.0      # Laplacian variance below this = blurry
MIN_DIM = 300              # reject images with shorter side < this
GRABCUT_ITERS = 5          # GrabCut refinement iterations
CROP_PAD = 20              # padding (px) around cropped leaf
ERODE_KERNEL = (7, 7)      # erosion kernel to strip black fringe
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Helper: is the file readable?
def readable(p):
    
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


# Helper: is this a studio shot?
def is_studio(img):
    h, w = img.shape[:2]

    # corner brightness
    c = max(1, int(min(h, w) * CORNER_FRAC))
    corners = [img[:c, :c], img[:c, -c:], img[-c:, :c], img[-c:, -c:]]
    if sum(1 for s in corners if s.mean() < BLACK_CORNER) >= MIN_BLACK_CORNERS:
        return True

    # border-strip black ratio
    gray = img.max(axis=2)
    if h < 2 * BORDER_PX or w < 2 * BORDER_PX:
        return False
    border = np.concatenate([
        gray[:BORDER_PX, :].ravel(), gray[-BORDER_PX:, :].ravel(),
        gray[:, :BORDER_PX].ravel(), gray[:, -BORDER_PX:].ravel(),
    ])
    return (border < BLACK_CORNER).mean() >= BORDER_RATIO


# Helper: check is the image blurry?
def is_blurry(img):
    """Laplacian variance: low value = soft/blurry image."""
    return cv2.Laplacian(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F
    ).var() < BLUR_THRESHOLD


# Helper: move a rejected file
def move_to_processed(f, cls, reason):
    """Move to data/processed/<reason>/<class>/ for manual review."""
    dest = PROCESSED_DIR / reason / cls
    dest.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(f), str(dest / f.name))
    except Exception as e:
        print(f"  [MOVE ERROR] {f.name}: {e}")


# Core: remove black studio background
def clean_studio_bg(img, out_path):
    """
    Steps:
      1. Find bounding box of non-black pixels (Region of Interest)
      2. Run GrabCut inside that box → binary mask
      3. Erode then dilate the mask (opening) → remove black fringe
      4. Crop to leaf bounding box
      5. Save as RGBA PNG (transparent background)
    """
    h, w = img.shape[:2]

    # bounding box of non-black pixels
    non_black = img.max(axis=2) >= BLACK_CORNER
    rows, cols = np.any(non_black, axis=1), np.any(non_black, axis=0)
    if not rows.any() or not cols.any():
        return False
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
    x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False

    # GrabCut segmentation
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1),
                    bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return False

    # Convert GrabCut labels to a binary alpha channel
    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # opening = erode (kills black ring) then dilate (restores leaf)
    alpha = cv2.erode(alpha, np.ones(ERODE_KERNEL, np.uint8), iterations=1)
    alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=1)

    # crop to leaf bounding box
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cx1, cy1 = max(0, bx - CROP_PAD), max(0, by - CROP_PAD)
    cx2, cy2 = min(w, bx + bw + CROP_PAD), min(h, by + bh + CROP_PAD)

    # merge BGR + alpha  RGBA, save PNG
    out = cv2.merge([*cv2.split(img[cy1:cy2, cx1:cx2]), alpha[cy1:cy2, cx1:cx2]])
    cv2.imwrite(str(out_path), out)
    return True


# Per-class pipeline
def process_class(cls):
    src, dst = RAW_DIR / cls, OUT_DIR / cls
    dst.mkdir(parents=True, exist_ok=True)

    # Counters for the final summary
    c = {"raw": 0, "corrupt": 0, "blurry": 0, "small": 0,
         "studio": 0, "outdoor": 0, "cleaned": 0, "copied": 0, "errors": 0}

    if not src.exists():
        return c

    files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in EXTS]
    c["raw"] = len(files)
    print(f"\n=== {cls} : {len(files)} raw images ===")

    for f in files:
        # Corrupt images delete
        if not readable(f):
            os.remove(f)
            c["corrupt"] += 1
            continue

        # Read image
        img = cv2.imread(str(f))
        if img is None:
            move_to_processed(f, cls, "error")
            c["errors"] += 1
            continue

        # Too small images processed/small
        if min(img.shape[:2]) < MIN_DIM:
            move_to_processed(f, cls, "small")
            c["small"] += 1
            continue

        # Studio shot images remove background
        if is_studio(img):
            c["studio"] += 1
            try:
                if clean_studio_bg(img, dst / (f.stem + ".png")):
                    c["cleaned"] += 1
                else:
                    move_to_processed(f, cls, "error")
                    c["errors"] += 1
            except Exception as e:
                print(f"  [ERROR] {f.name}: {e}")
                move_to_processed(f, cls, "error")
                c["errors"] += 1
            continue

        # Blurry non-studio  processed/blurry
        if is_blurry(img):
            move_to_processed(f, cls, "blurry")
            c["blurry"] += 1
            continue

        # Outdoor, sharp just copy no change
        c["outdoor"] += 1
        try:
            cv2.imwrite(str(dst / (f.stem + ".png")), img)
            c["copied"] += 1
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            move_to_processed(f, cls, "error")
            c["errors"] += 1

    return c


# Entry point
def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} missing")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Run the pipeline for each class
    stats = {cls: process_class(cls) for cls in CLASSES}

    # Summary table
    print("\n========== SUMMARY ==========")
    hdr = (f"{'Class':<16}{'Raw':>6}{'Corrupt':>9}{'Blurry':>8}{'Small':>7}"
           f"{'Studio':>8}{'Outdoor':>9}{'Cleaned':>9}{'Copied':>8}{'Errors':>8}")
    print(hdr)
    print("-" * len(hdr))
    for cls, s in stats.items():
        print(f"{cls:<16}{s['raw']:>6}{s['corrupt']:>9}{s['blurry']:>8}"
              f"{s['small']:>7}{s['studio']:>8}{s['outdoor']:>9}"
              f"{s['cleaned']:>9}{s['copied']:>8}{s['errors']:>8}")

    print(f"\nCleaned  : {OUT_DIR}")
    print(f"Processed: {PROCESSED_DIR}  (blurry / small / error)")


if __name__ == "__main__":
    main()