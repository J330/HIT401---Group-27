"""Render a colour heatmap restricted to the U2Net leaf silhouette.

The original Grad-CAM is not a lesion segmentation. The mask is used purely
for visual compositing and never alters disease classification or attention.
"""
import base64
import io
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.nn import functional as F

MAX_SIDE = 900
ENHANCE_FLOOR = 0.03
ENHANCE_GAMMA = 0.60
ENHANCE_PERCENTILE = 0.95
# Display-only expansion of high-attention regions (not a lesion boundary).
HOTSPOT_SEED_THRESHOLD = 0.74
HOTSPOT_RADIUS_FRACTION = 0.018  # about 1.8% of the shorter image side
HOTSPOT_STRENGTH = 1.16


def _data_url(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def _enhance_cam(cam: np.ndarray, foreground_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Boost moderate regions so the viewer shows warm colours more readily.

    This is a display remap only. It never fabricates a hotspot when the raw CAM
    is flat, but it does spread the upper-middle range so moderately important
    areas can become orange/red instead of staying mostly cyan/yellow.
    """
    cam = np.clip(np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    if foreground_mask is not None:
        if foreground_mask.shape != cam.shape:
            raise ValueError("Foreground mask must match CAM dimensions")
        visible = foreground_mask > 0.10
        if not np.any(visible):
            return np.zeros_like(cam)
        measured = cam[visible]
    else:
        visible = None
        measured = cam
    spread = float(measured.max() - measured.min())
    if spread > 1e-6:
        cam = (cam - measured.min()) / spread
    else:
        # Do not invent a hotspot when the foreground has uniform attribution.
        return np.zeros_like(cam)
    if visible is not None:
        cam = np.where(visible, cam, 0.0)

    # Treat the upper-middle of the real attention range as visually strong so
    # the heatmap becomes more sensitive and reveals warmer colours earlier.
    # Percentile normalisation still preserves ordering and avoids fabricating
    # non-existent hotspots in very flat maps.
    hot = float(np.quantile(cam[visible] if visible is not None else cam, ENHANCE_PERCENTILE))
    if hot > 1e-6:
        cam = np.clip(cam / hot, 0.0, 1.0)

    cam = np.clip((cam - ENHANCE_FLOOR) / max(1e-6, 1.0 - ENHANCE_FLOOR), 0.0, 1.0)
    cam = np.power(cam, ENHANCE_GAMMA)
    return np.clip(cam, 0.0, 1.0)


def _expand_hotspots(cam: np.ndarray, foreground_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Visually widen *existing* high-activation areas, without detecting lesions.

    The operation seeds only from high attribution values and limits its
    spatial reach to a small neighbourhood; unrelated low-value areas stay
    blue. This map is for visualization only and must not be treated as
    quantitative attribution or a disease-segmentation mask.
    """
    height, width = cam.shape
    if foreground_mask is not None:
        if foreground_mask.shape != cam.shape:
            raise ValueError("Foreground mask must match CAM dimensions")
        valid = foreground_mask > 0.10
        cam = np.where(valid, cam, 0.0)
    else:
        valid = None
    radius = max(1, min(18, round(min(height, width) * HOTSPOT_RADIUS_FRACTION)))
    seed = torch.as_tensor(
        np.where(cam >= HOTSPOT_SEED_THRESHOLD, cam, 0.0).astype(np.float32)
    )[None, None]
    if not torch.any(seed):
        return cam

    kernel = radius * 2 + 1
    expanded = F.max_pool2d(seed, kernel, stride=1, padding=radius)
    # Keep transitions gradual rather than showing hard-edged dilated circles.
    softened = F.avg_pool2d(expanded, kernel_size=3, stride=1, padding=1)
    expanded = (softened * HOTSPOT_STRENGTH).clamp(0.0, 1.0)
    original = torch.as_tensor(cam, dtype=torch.float32)[None, None]
    expanded_result = torch.maximum(original, expanded)[0, 0].numpy().clip(0.0, 1.0)
    # Dilation must never escape the foreground, even when it starts near an edge.
    return np.where(valid, expanded_result, 0.0) if valid is not None else expanded_result


def _display_mask(foreground_mask: Optional[Image.Image], image_size: Tuple[int, int],
                  output_size: Tuple[int, int]) -> np.ndarray:
    """Aligned, softly feathered display alpha; zero outside detected leaf.

    Mask resizing uses the same whole-frame transformation as the heatmap and
    photo. Tiny alpha noise is suppressed; a gentle inward edge avoids bright
    halos around the isolated leaf. A missing mask means old full-frame mode.
    """
    if foreground_mask is None:
        return np.ones((output_size[1], output_size[0]), dtype=np.float32)
    if foreground_mask.size != image_size:
        raise ValueError("Foreground mask and isolated image must have matching sizes")

    raw = foreground_mask.convert('L').resize(output_size, Image.Resampling.BILINEAR)
    # Suppress faint removal artifacts, then gently taper the visible edge.
    raw = raw.point(lambda x: int(max(0.0, min(255.0, (x - 24) * (255 / 231)))))
    # Feather INWARD: blurring alone would leak colour beyond the silhouette.
    support = np.asarray(raw, dtype=np.uint8) > 0
    raw = raw.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.1))
    values = np.asarray(raw, dtype=np.float32) / 255.0
    return np.where(support, np.clip(values, 0.0, 1.0), 0.0)


def make_heatmap_urls(image: Image.Image, grayscale_cam,
                      foreground_mask: Optional[Image.Image] = None):
    """Return (RGBA attention PNG URL, masked colour overlay PNG URL).

    RGB channels of the first image hold the enhanced grayscale attention;
    its alpha channel holds the foreground mask, consumed by result.js.
    A missing mask (including rembg fallback) preserves full-frame rendering.
    The isolated image, mask and CAM must remain spatially aligned.
    """
    source = ImageOps.exif_transpose(image).convert('RGB')
    original_size = source.size
    source.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    display_mask = _display_mask(foreground_mask, original_size, source.size)
    cam = torch.as_tensor(np.asarray(grayscale_cam).copy(), dtype=torch.float32)[None, None]
    cam = F.interpolate(cam, size=(source.height, source.width), mode='bilinear', align_corners=False)[0, 0].numpy()
    # Build high-intensity regions from valid leaf pixels only. The mask then
    # clips even expanded hotspots to the same leaf silhouette.
    enhanced = _enhance_cam(cam, display_mask if foreground_mask is not None else None)
    enhanced = _expand_hotspots(enhanced, display_mask if foreground_mask is not None else None)
    cam = enhanced
    if foreground_mask is not None:
        cam = np.where(display_mask > 0.02, cam, 0.0)

    grayscale = np.uint8(np.round(cam * 255))
    display_alpha = np.uint8(np.round(display_mask * 255))
    heatmap_rgba = Image.fromarray(np.dstack((grayscale, grayscale, grayscale, display_alpha)), 'RGBA')

    # A rainbow/"jet"-style HOTSPOT ramp matching the browser viewer.
    # These are class-attribution intensities, not disease segmentation labels.
    stops = np.array([
        [0.00,  10,  45, 185],   # blue: lowest relative attribution
        [0.24,  20, 105, 240],   # bright blue
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
    # Keep blue-to-red inside the leaf. Completely transparent over removed
    # background, including any expanded hotspots. At boundaries the mask
    # provides a gentle fade into the neutral-grey isolated image.
    alpha = (0.58 + 0.29 * np.sqrt(cam)) * display_mask
    coloured.putalpha(Image.fromarray(np.uint8(np.round(alpha * 255)), 'L'))
    overlay = Image.alpha_composite(source.convert('RGBA'), coloured).convert('RGB')
    return _data_url(heatmap_rgba), _data_url(overlay)
