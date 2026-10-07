# ml/clean_data.py


# References

# https://www.geeksforgeeks.org/image-blur-detection-using-opencv/

# https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html

# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify

# https://en.wikipedia.org/wiki/RGBA_color_model

# ml/clean_data.py
#
# Non-destructive banana leaf dataset cleaner for outdoor images.
# Raw images are never modified — only read. Cleaned outputs are written
# to data/cleaned/, and rejects are copied to data/processed/.
#
# Pipeline per class:
#   1. Corrupt files     → copied to processed/corrupt/
#   2. Duplicates        → copied to processed/duplicates/
#   3. Small (<MIN_DIM)  → copied to processed/small/
#   4. Over-bright/dark  → copied to processed/exposure/
#   5. Blurry            → copied to processed/blurry/
#   6. Leaf too small    → zoom-cropped, saved to cleaned/
#   7. Remaining         → saved to cleaned/
#
# Install:  pip install opencv-python numpy pillow imagehash
#
# References:
#   PHash dedup:     https://github.com/JohannesBuchner/imagehash
#   Pillow verify:   https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify
#   HSV colorspace:  https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#   Blur (Laplacian): Pech-Pacheco et al. (2000), ICPR.

import shutil
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "cleaned"
PROCESSED_DIR = ROOT / "data" / "processed"
CLASSES = ["healthy", "black_sigatoka"]

# ---------- Tunables ----------
MIN_DIM = 300
BLUR_THRESHOLD = 15.0
DEDUP_THRESHOLD = 10

DARK_MEAN_MAX = 40
BRIGHT_MEAN_MIN = 220
LOW_CONTRAST_STD = 15

LEAF_AREA_MIN_RATIO = 0.10
ZOOM_PAD = 20

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------- Helpers ----------
def readable(p):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def is_blurry(img):
    return cv2.Laplacian(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F
    ).var() < BLUR_THRESHOLD


def exposure_status(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = gray.mean(), gray.std()
    if mean < DARK_MEAN_MAX:  return "dark"
    if mean > BRIGHT_MEAN_MIN: return "bright"
    if std < LOW_CONTRAST_STD: return "flat"
    return "ok"


def find_duplicates(folder_path, threshold=DEDUP_THRESHOLD):
    hashes = {}
    for f in sorted(folder_path.iterdir()):
        if f.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(f) as im:
                hashes[f.name] = imagehash.phash(im)
        except Exception:
            continue

    names = list(hashes)
    dup_map = {}
    claimed = set()
    for i in range(len(names)):
        canon = names[i]
        if canon in claimed:
            continue
        for j in range(i + 1, len(names)):
            cand = names[j]
            if cand in claimed:
                continue
            if hashes[canon] - hashes[cand] <= threshold:
                dup_map.setdefault(canon, []).append(cand)
                claimed.add(cand)
    return dup_map


def leaf_bbox(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def zoom_crop(img):
    bbox = leaf_bbox(img)
    if bbox is None:
        return img
    h, w = img.shape[:2]
    bx, by, bw, bh = bbox
    if (bw * bh) / float(w * h) >= LEAF_AREA_MIN_RATIO:
        return img
    x1 = max(0, bx - ZOOM_PAD)
    y1 = max(0, by - ZOOM_PAD)
    x2 = min(w, bx + bw + ZOOM_PAD)
    y2 = min(h, by + bh + ZOOM_PAD)
    return img[y1:y2, x1:x2]


def copy_to_processed(f, cls, reason):
    """COPY (not move) a rejected file to processed/<reason>/<class>/."""
    dest_dir = PROCESSED_DIR / reason / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f.name
    if dest.exists():
        # avoid clobbering, add a suffix
        dest = dest_dir / (f.stem + "_dup" + f.suffix)
    try:
        shutil.copy2(str(f), str(dest))
    except Exception as e:
        print(f"  [COPY ERROR] {f.name}: {e}")


# ---------- Per-class pipeline ----------
def process_class(cls):
    src = RAW_DIR / cls
    dst = OUT_DIR / cls
    dst.mkdir(parents=True, exist_ok=True)

    c = {"raw": 0, "corrupt": 0, "duplicates": 0, "small": 0,
         "dark": 0, "bright": 0, "flat": 0, "blurry": 0,
         "zoomed": 0, "saved": 0, "errors": 0}

    if not src.exists():
        return c

    files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in EXTS]
    c["raw"] = len(files)
    print(f"\n=== {cls} : {len(files)} raw images ===")

    # Step 1: check readable — copy corrupt ones, don't delete from raw
    valid = []
    for f in files:
        if readable(f):
            valid.append(f)
        else:
            copy_to_processed(f, cls, "corrupt")
            c["corrupt"] += 1
    if c["corrupt"]:
        print(f"  found {c['corrupt']} corrupt file(s)")

    # Step 2: find duplicates among valid files
    print(f"  scanning for duplicates...")
    # only hash valid files — avoid trying to open corrupt ones
    hashes = {}
    for f in valid:
        try:
            with Image.open(f) as im:
                hashes[f.name] = imagehash.phash(im)
        except Exception:
            continue
    names = list(hashes)
    dup_names = set()
    claimed = set()
    for i in range(len(names)):
        canon = names[i]
        if canon in claimed:
            continue
        for j in range(i + 1, len(names)):
            cand = names[j]
            if cand in claimed:
                continue
            if hashes[canon] - hashes[cand] <= DEDUP_THRESHOLD:
                dup_names.add(cand)
                claimed.add(cand)
    for d in dup_names:
        copy_to_processed(src / d, cls, "duplicates")
    c["duplicates"] = len(dup_names)
    print(f"  found {c['duplicates']} duplicate(s)")

    # Steps 3–7: process remaining
    for f in valid:
        if f.name in dup_names:
            continue

        img = cv2.imread(str(f))
        if img is None:
            copy_to_processed(f, cls, "error")
            c["errors"] += 1
            continue

        if min(img.shape[:2]) < MIN_DIM:
            copy_to_processed(f, cls, "small")
            c["small"] += 1
            continue

        status = exposure_status(img)
        if status in ("dark", "bright", "flat"):
            copy_to_processed(f, cls, "exposure")
            c[status] += 1
            continue

        if is_blurry(img):
            copy_to_processed(f, cls, "blurry")
            c["blurry"] += 1
            continue

        cropped = zoom_crop(img)
        if cropped.shape != img.shape:
            c["zoomed"] += 1

        cv2.imwrite(str(dst / (f.stem + ".png")), cropped)
        c["saved"] += 1

    return c


# ---------- Entry point ----------
def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} missing")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = {cls: process_class(cls) for cls in CLASSES}

    print("\n========== SUMMARY ==========")
    hdr = (f"{'Class':<16}{'Raw':>6}{'Corrupt':>9}{'Dup':>6}{'Small':>7}"
           f"{'Dark':>6}{'Bright':>8}{'Flat':>6}{'Blur':>6}"
           f"{'Zoom':>6}{'Saved':>7}{'Err':>5}")
    print(hdr)
    print("-" * len(hdr))
    for cls, s in stats.items():
        print(f"{cls:<16}{s['raw']:>6}{s['corrupt']:>9}{s['duplicates']:>6}"
              f"{s['small']:>7}{s['dark']:>6}{s['bright']:>8}{s['flat']:>6}"
              f"{s['blurry']:>6}{s['zoomed']:>6}{s['saved']:>7}{s['errors']:>5}")

    print(f"\nCleaned  : {OUT_DIR}")
    print(f"Processed: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()