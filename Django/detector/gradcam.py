"""Explainability utilities for the DINOv2 health-classification stage.

DINOv2 is a Vision Transformer, so a classic CNN-style Grad-CAM overlay can be
very coarse and noisy. This module instead uses class-specific gradient ×
activation attribution on the final DINOv2 patch grid, keeps only the most
influential patches, and (when available) suppresses pixels outside the
foreground leaf mask produced by the existing rembg pipeline.

The result is an explanation of which image regions most influenced the chosen
class. It is not a disease-segmentation mask and must not be interpreted as an
exact outline of lesions.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps


MAX_HEATMAP_SIDE = 900
OVERLAY_ALPHA = 0.80
TOP_IMPORTANCE_FRACTION = 0.30


def _first_tensor(value: Any) -> torch.Tensor | None:
    """Return the first tensor contained in a layer output."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _target_layer(runtime: Any) -> torch.nn.Module:
    """Choose a late DINOv2 transformer layer that still retains patch tokens."""
    if runtime.key != "dinov2-base" or runtime.model_type != "timm_dinov2":
        raise RuntimeError("This heatmap implementation only supports DINOv2 Base.")

    model = runtime.model
    blocks = getattr(model, "blocks", None)
    if blocks:
        last_block = blocks[-1]
        norm1 = getattr(last_block, "norm1", None)
        if norm1 is not None:
            return norm1
        return last_block

    norm = getattr(model, "norm", None)
    if norm is not None:
        return norm

    raise RuntimeError("No suitable DINOv2 transformer layer was found.")


def _tokens_to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a transformer token tensor to [B, C, H, W]."""
    if tensor.ndim != 3:
        raise RuntimeError(f"Expected a 3D token tensor, got shape {tuple(tensor.shape)}")

    batch, dim1, dim2 = tensor.shape

    def grid_size(token_count: int) -> tuple[int, int] | None:
        # DINOv2 normally has one class token. Some timm variants can also have
        # register/prefix tokens, so accept a small number of leading tokens.
        for prefix_tokens in range(0, 9):
            spatial_tokens = token_count - prefix_tokens
            if spatial_tokens <= 0:
                continue
            root = int(math.sqrt(spatial_tokens))
            if root * root == spatial_tokens:
                return root, prefix_tokens
        return None

    token_layout = grid_size(dim1)
    if token_layout is not None:
        side, prefix_tokens = token_layout
        tokens = tensor[:, prefix_tokens:, :]
        return tokens.transpose(1, 2).reshape(batch, dim2, side, side)

    # Fallback for uncommon [B, C, tokens] layouts.
    token_layout = grid_size(dim2)
    if token_layout is not None:
        side, prefix_tokens = token_layout
        tokens = tensor[:, :, prefix_tokens:]
        return tokens.reshape(batch, dim1, side, side)

    raise RuntimeError(
        "Could not infer the transformer patch grid from activation shape "
        f"{tuple(tensor.shape)}."
    )


def _as_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise activation/gradient layout to [B, C, H, W]."""
    if tensor.ndim == 4:
        if tensor.shape[1] <= 32 and tensor.shape[-1] > 32:
            return tensor.permute(0, 3, 1, 2)
        return tensor
    if tensor.ndim == 3:
        return _tokens_to_spatial(tensor)
    raise RuntimeError(
        f"Explainability target layer returned unsupported shape {tuple(tensor.shape)}."
    )


def _forward_for_attribution(image: Image.Image, runtime: Any) -> torch.Tensor:
    if runtime.key != "dinov2-base" or runtime.model_type != "timm_dinov2":
        raise RuntimeError("This heatmap implementation only supports DINOv2 Base.")

    device = next(runtime.model.parameters()).device
    tensor = runtime.processor_or_transform(image).unsqueeze(0).to(device)
    return runtime.model(tensor)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Convert a 0..1 grayscale array into an RGB heatmap without OpenCV."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def _resize_for_display(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) <= MAX_HEATMAP_SIDE:
        return image.copy()

    resized = image.copy()
    resized.thumbnail((MAX_HEATMAP_SIDE, MAX_HEATMAP_SIDE), Image.Resampling.LANCZOS)
    return resized


def _robust_normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """Normalise attribution while preventing a few outliers flattening the map."""
    array = cam.detach().float().cpu().numpy()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise RuntimeError("The heatmap contained no finite values.")

    low = float(np.percentile(finite, 5.0))
    high = float(np.percentile(finite, 97.0))

    if high - low < 1e-12:
        low = float(finite.min())
        high = float(finite.max())

    if high - low < 1e-12:
        raise RuntimeError("The model produced a flat attention map for this image.")

    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return torch.from_numpy(array.astype(np.float32))


def _keep_top_regions(cam_array: np.ndarray) -> np.ndarray:
    """Suppress weak/noisy regions and retain roughly the top 30% influence."""
    values = np.asarray(cam_array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    keep_fraction = float(np.clip(TOP_IMPORTANCE_FRACTION, 0.05, 1.0))
    threshold = float(np.quantile(positive, 1.0 - keep_fraction))
    high = float(positive.max())

    if high - threshold < 1e-8:
        threshold = float(np.quantile(positive, 0.50))

    focused = np.clip((values - threshold) / max(high - threshold, 1e-8), 0.0, 1.0)

    # Slight gamma lift keeps strong secondary patches visible without painting
    # the low-importance majority of the image blue/cyan.
    focused = np.power(focused, 0.72)
    return focused.astype(np.float32)


def _leaf_foreground_mask(image: Image.Image, size: tuple[int, int]) -> np.ndarray | None:
    """Return a soft leaf/foreground mask using the already-cached rembg session.

    If rembg cannot produce a trustworthy mask, return None and leave the class
    attribution unchanged. This keeps explainability available even when
    foreground isolation fails.
    """
    try:
        # Imported lazily to avoid making explainability depend on rembg startup
        # unless the heatmap is actually requested.
        from .background_filter import _get_rembg_session, _to_rgba_removed

        source = ImageOps.exif_transpose(image).convert("RGB")
        cutout = _to_rgba_removed(source, _get_rembg_session())
        alpha = cutout.getchannel("A")

        mask_array = np.asarray(alpha, dtype=np.float32) / 255.0
        coverage = float(np.mean(mask_array >= (24.0 / 255.0)))
        if coverage < 0.015 or coverage > 0.985:
            return None

        # A small blur avoids a harsh cut-out edge in the attribution overlay.
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2.0))
        alpha = alpha.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(alpha, dtype=np.float32) / 255.0
    except Exception:
        return None


def _overlay_png_data_url(image: Image.Image, cam: torch.Tensor) -> str:
    base = _resize_for_display(image)

    # First enlarge the patch attribution to display resolution.
    cam = cam.detach().float().cpu()[None, None, :, :]
    cam = F.interpolate(
        cam,
        size=(base.height, base.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam_array = np.clip(cam.numpy(), 0.0, 1.0)

    # Keep only the strongest regions. This is the main visual cleanup versus
    # the previous full-frame Grad-CAM overlay.
    focused = _keep_top_regions(cam_array)

    # Where the existing foreground model is trustworthy, stop background
    # foliage/sky from dominating the explanation.
    foreground = _leaf_foreground_mask(image, base.size)
    if foreground is not None:
        focused *= np.power(np.clip(foreground, 0.0, 1.0), 1.35)

    maximum = float(focused.max())
    if maximum > 1e-8:
        focused = focused / maximum

    rgb = (_jet_colormap(focused) * 255).astype(np.uint8)
    heatmap = Image.fromarray(rgb, mode="RGB")

    # Fully suppress weak patches rather than tinting the whole image blue.
    alpha_strength = np.where(focused > 0.0, np.power(focused, 0.62), 0.0)
    alpha = np.clip(alpha_strength * OVERLAY_ALPHA, 0.0, OVERLAY_ALPHA)
    alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    heatmap.putalpha(alpha_image)

    composite = Image.alpha_composite(base.convert("RGBA"), heatmap).convert("RGB")

    buffer = io.BytesIO()
    composite.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _generate_patch_attribution(
    image: Image.Image,
    runtime: Any,
    class_index: int,
) -> tuple[str, str]:
    """Generate class-specific gradient × activation attribution per DINO patch."""
    model = runtime.model
    target_layer = _target_layer(runtime)
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        tensor = _first_tensor(output)
        if tensor is None:
            return
        captured["activations"] = tensor
        if tensor.requires_grad:
            tensor.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = _forward_for_attribution(image, runtime)
            if logits.ndim != 2 or class_index < 0 or class_index >= logits.shape[1]:
                raise RuntimeError("The predicted class index is invalid for attribution.")

            logits[0, class_index].backward()

        activations = captured.get("activations")
        gradients = captured.get("gradients")
        if activations is None or gradients is None:
            raise RuntimeError("The target layer did not expose activations and gradients.")

        activations = _as_spatial(activations.detach())
        gradients = _as_spatial(gradients.detach())

        if activations.shape != gradients.shape:
            raise RuntimeError(
                "Attribution activation and gradient shapes did not match: "
                f"{tuple(activations.shape)} vs {tuple(gradients.shape)}"
            )

        # A signed gradient × activation sum estimates positive class-specific
        # contribution for each transformer patch. It retains locality better
        # than averaging gradients into one global channel weight as Grad-CAM does.
        contribution = (gradients * activations).sum(dim=1)[0]
        cam = torch.relu(contribution)

        if float((cam.max() - cam.min()).abs().item()) < 1e-12:
            # Fallback for classes whose signed contributions cancel.
            cam = (gradients * activations).abs().mean(dim=1)[0]

        cam = _robust_normalize_cam(cam)
        return _overlay_png_data_url(image, cam), "DINOv2 patch attribution"
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _generate_input_gradient_data_url(
    image: Image.Image,
    runtime: Any,
    class_index: int,
) -> tuple[str, str]:
    """Fallback class-specific saliency map based on input gradient × image."""
    model = runtime.model
    device = next(model.parameters()).device
    model.eval()
    model.zero_grad(set_to_none=True)

    if runtime.key != "dinov2-base" or runtime.model_type != "timm_dinov2":
        raise RuntimeError("This saliency implementation only supports DINOv2 Base.")

    with torch.enable_grad():
        input_tensor = runtime.processor_or_transform(image).unsqueeze(0).to(device).detach()
        input_tensor.requires_grad_(True)
        logits = model(input_tensor)

        if logits.ndim != 2 or class_index < 0 or class_index >= logits.shape[1]:
            raise RuntimeError("The predicted class index is invalid for saliency mapping.")

        logits[0, class_index].backward()

        gradient = input_tensor.grad
        if gradient is None:
            raise RuntimeError("The model did not expose an input gradient.")

        saliency = (gradient * input_tensor).abs().mean(dim=1, keepdim=True)
        if saliency.shape[-2] >= 3 and saliency.shape[-1] >= 3:
            saliency = F.avg_pool2d(saliency, kernel_size=5, stride=1, padding=2)
        cam = saliency[0, 0]

        if float((cam.max() - cam.min()).abs().item()) < 1e-12:
            cam = gradient.abs().mean(dim=1)[0]

        cam = _robust_normalize_cam(cam)
        return _overlay_png_data_url(image, cam), "Input gradient × image"


def generate_gradcam_data_url(
    image: Image.Image,
    runtime: Any,
    class_index: int,
) -> tuple[str, str]:
    """Return a focused class-specific DINOv2 explanation overlay.

    Patch attribution is preferred. If it cannot be produced for a particular
    image/model state, fall back to input-gradient saliency rather than removing
    explainability from the result page entirely.
    """
    primary_error_text = "Unknown patch-attribution error"
    try:
        return _generate_patch_attribution(image, runtime, class_index)
    except Exception as error:
        primary_error_text = str(error)

    try:
        return _generate_input_gradient_data_url(image, runtime, class_index)
    except Exception as fallback_error:
        raise RuntimeError(
            "DINOv2 patch attribution and the fallback saliency method both failed. "
            f"Patch attribution: {primary_error_text}. Fallback: {fallback_error}."
        ) from fallback_error
