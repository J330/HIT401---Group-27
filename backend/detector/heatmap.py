"""Render truthful class-attribution maps in a high-contrast blue/cyan/yellow/red hotspot palette.

Keep both the grayscale CAM (for the interactive frontend) and a blended
overlay (API consumers). Color alone cannot improve model spatial resolution.
"""
import base64
import io
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

MAX_SIDE = 900
ENHANCE_FLOOR = 0.12
ENHANCE_GAMMA = 0.95


def _data_url(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def _enhance_cam(cam: np.ndarray) -> np.ndarray:
    """Boost the strongest regions so the viewer highlights them more clearly."""
    cam = np.clip(np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    spread = float(cam.max() - cam.min())
    if spread > 1e-6:
        cam = (cam - cam.min()) / spread
    else:
        # Do not invent a hotspot when the classifier returns a flat CAM.
        cam = np.zeros_like(cam)
    cam = np.clip((cam - ENHANCE_FLOOR) / max(1e-6, 1.0 - ENHANCE_FLOOR), 0.0, 1.0)
    cam = np.power(cam, ENHANCE_GAMMA)
    return np.clip(cam, 0.0, 1.0)


def make_heatmap_urls(image: Image.Image, grayscale_cam):
    """Return (grayscale PNG URL, colour overlay PNG URL).

    The grayscale image is used by result.js for the interactive viewer. We
    emphasise peak attention and generate a visible hotspot overlay.
    We do NOT detect, trace or invent lesions: all colour comes from model-derived attribution.
    """
    source = ImageOps.exif_transpose(image).convert('RGB')
    source.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    cam = torch.as_tensor(np.asarray(grayscale_cam).copy(), dtype=torch.float32)[None, None]
    cam = F.interpolate(cam, size=(source.height, source.width), mode='bilinear', align_corners=False)[0, 0].numpy()
    cam = _enhance_cam(cam)
    gray = Image.fromarray(np.uint8(cam * 255), 'L')

    # A rainbow/"jet"-style HOTSPOT ramp matching the browser viewer.
    # These are class-attribution intensities, not disease segmentation labels.
    stops = np.array([
        [0.00,   8,  24,  97],   # deep blue (normally transparent)
        [0.24,  10,  76, 222],   # blue edges
        [0.37,   0, 190, 255],   # cyan rims
        [0.51,  30, 245, 183],   # turquoise
        [0.64, 255, 238,  28],   # yellow
        [0.78, 255, 131,   8],   # orange
        [0.91, 255,  29,  12],   # red hotspot
        [1.00, 196,   0,   0],   # dark red core
    ], dtype=np.float32)
    red = np.interp(cam, stops[:, 0], stops[:, 1])
    green = np.interp(cam, stops[:, 0], stops[:, 2])
    blue = np.interp(cam, stops[:, 0], stops[:, 3])
    coloured = Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8), 'RGB').convert('RGBA')
    # Keep the leaf legible. Weak regions disappear, while cyan rims and red
    # peaks are intense. Alpha strength matches result.js at 100% opacity.
    strength = np.clip((cam - 0.19) / 0.76, 0, 1)
    strength = strength * strength * (3 - 2 * strength)
    alpha = np.where(cam < 0.19, 0, 0.10 + 0.85 * strength)
    coloured.putalpha(Image.fromarray(np.uint8(alpha * 255), 'L'))
    overlay = Image.alpha_composite(source.convert('RGBA'), coloured).convert('RGB')
    return _data_url(gray), _data_url(overlay)
