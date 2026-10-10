"""Reusable foreground isolation from the original HIT401 Django project.

This module removes background via rembg/u2netp. In this integration it is
automatic preprocessing only for Grad-CAM, after Stage 1 and Stage 2
have already evaluated the original photograph.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# u2netp is the small rembg model and is sufficient for a simple foreground
# isolation pass. rembg downloads it once on first use and caches it locally.
REMBG_MODEL_NAME = "u2netp"
NEUTRAL_BACKGROUND_RGB = (128, 128, 128)
CROP_PADDING_RATIO = 0.08


@dataclass
class BackgroundFilterResult:
    image: Image.Image
    applied: bool
    method: str
    reason: str | None = None
    # The original rembg alpha, cropped identically to image. Keep it for
    # display-only masking of the heatmap; never alter classifier inputs.
    foreground_mask: Image.Image | None = None


_SESSION_LOCK = threading.Lock()
_REMBG_SESSION: Any | None = None
_REMBG_DISABLED_REASON: str | None = None


def _get_rembg_session() -> Any:
    global _REMBG_SESSION, _REMBG_DISABLED_REASON

    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    if _REMBG_DISABLED_REASON is not None:
        raise RuntimeError(_REMBG_DISABLED_REASON)

    with _SESSION_LOCK:
        if _REMBG_SESSION is not None:
            return _REMBG_SESSION

        try:
            from rembg import new_session

            _REMBG_SESSION = new_session(REMBG_MODEL_NAME)
            return _REMBG_SESSION
        except Exception as error:
            _REMBG_DISABLED_REASON = (
                "Background removal could not start. Install the requirements "
                "and make sure the rembg model can be downloaded on first use."
            )
            raise RuntimeError(_REMBG_DISABLED_REASON) from error


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size

    object_width = max(1, right - left)
    object_height = max(1, bottom - top)
    pad_x = max(2, round(object_width * padding_ratio))
    pad_y = max(2, round(object_height * padding_ratio))

    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def _to_rgba_removed(image: Image.Image, session: Any) -> Image.Image:
    """Call rembg in a way that works across PIL/bytes return variants."""
    from rembg import remove

    source = image.convert("RGB")
    result = remove(source, session=session)

    if isinstance(result, Image.Image):
        return result.convert("RGBA")

    if isinstance(result, (bytes, bytearray)):
        with Image.open(io.BytesIO(result)) as decoded:
            decoded.load()
            return decoded.convert("RGBA")

    raise RuntimeError("rembg returned an unsupported image format.")


def make_background_filtered_view(image: Image.Image) -> BackgroundFilterResult:
    """Return a cropped foreground image on a neutral-grey background.

    Any failure falls back to the original image so the scan remains usable.
    """
    original = image.convert("RGB")

    try:
        session = _get_rembg_session()
        cutout = _to_rgba_removed(original, session)
        alpha = cutout.getchannel("A")

        # Ignore very faint alpha noise when finding the foreground bounds.
        solid_mask = alpha.point(lambda value: 255 if value >= 24 else 0)
        bbox = solid_mask.getbbox()
        if bbox is None:
            raise RuntimeError("The foreground filter did not find a usable object.")

        mask_pixels = solid_mask.histogram()[255]
        total_pixels = max(1, solid_mask.width * solid_mask.height)
        coverage = mask_pixels / total_pixels

        # Extremely tiny/full masks usually mean segmentation failed. Returning
        # the original view is safer than generating a misleading attention view.
        if coverage < 0.015 or coverage > 0.985:
            raise RuntimeError(
                f"Foreground mask coverage looked unreliable ({coverage:.1%})."
            )

        neutral = Image.new("RGB", cutout.size, NEUTRAL_BACKGROUND_RGB)
        neutral.paste(cutout.convert("RGB"), mask=alpha)

        padded_bbox = _expand_bbox(bbox, neutral.size, CROP_PADDING_RATIO)
        filtered = neutral.crop(padded_bbox)
        foreground_mask = alpha.crop(padded_bbox)

        if filtered.width < 16 or filtered.height < 16:
            raise RuntimeError("The isolated foreground crop was too small.")

        return BackgroundFilterResult(
            image=filtered,
            applied=True,
            method=f"rembg:{REMBG_MODEL_NAME}",
            foreground_mask=foreground_mask,
        )

    except Exception as error:
        logger.warning("Background removal was skipped: %s", error)
        return BackgroundFilterResult(
            image=original,
            applied=False,
            method="original-only fallback",
            reason=str(error),
        )
