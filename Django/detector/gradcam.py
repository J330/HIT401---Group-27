"""Grad-CAM utilities for the health-classification stage.

The implementation is configured specifically for this project's fine-tuned
timm DINOv2 Base model.

It returns a small PNG data URL so Django does not need to save temporary
heatmap files to disk or configure a MEDIA_ROOT just for explainability.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


MAX_HEATMAP_SIDE = 900
OVERLAY_ALPHA = 0.76


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


def _last_conv2d(module: torch.nn.Module) -> torch.nn.Module | None:
    candidate = None
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            candidate = child
    return candidate


def _target_layer(runtime: Any) -> torch.nn.Module:
    """Choose the final DINOv2 transformer attention block for Grad-CAM."""
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

    raise RuntimeError("No suitable DINOv2 transformer layer was found for Grad-CAM.")


def _tokens_to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, tokens, channels] transformer output to [B, C, H, W]."""
    if tensor.ndim != 3:
        raise RuntimeError(f"Expected a 3D token tensor, got shape {tuple(tensor.shape)}")

    batch, dim1, dim2 = tensor.shape

    def grid_size(token_count: int) -> tuple[int, bool] | None:
        root = int(math.sqrt(token_count))
        if root * root == token_count:
            return root, False
        root = int(math.sqrt(max(0, token_count - 1)))
        if root * root == token_count - 1:
            return root, True
        return None

    # DINOv2 is normally [B, 1 + H*W, C]. Prefer this interpretation first.
    token_layout = grid_size(dim1)
    if token_layout is not None:
        side, has_cls = token_layout
        tokens = tensor[:, 1:, :] if has_cls else tensor
        return tokens.transpose(1, 2).reshape(batch, dim2, side, side)

    # Fallback for uncommon [B, C, tokens] layouts.
    token_layout = grid_size(dim2)
    if token_layout is not None:
        side, has_cls = token_layout
        tokens = tensor[:, :, 1:] if has_cls else tensor
        return tokens.reshape(batch, dim1, side, side)

    raise RuntimeError(
        "Could not infer the transformer patch grid from activation shape "
        f"{tuple(tensor.shape)}."
    )


def _as_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise activation/gradient layout to [B, C, H, W]."""
    if tensor.ndim == 4:
        # Most CNNs are channels-first. If the last dimension is clearly the
        # channel dimension, convert channels-last to channels-first.
        if tensor.shape[1] <= 32 and tensor.shape[-1] > 32:
            return tensor.permute(0, 3, 1, 2)
        return tensor
    if tensor.ndim == 3:
        return _tokens_to_spatial(tensor)
    raise RuntimeError(
        f"Grad-CAM target layer returned unsupported shape {tuple(tensor.shape)}."
    )


def _forward_for_gradcam(image: Image.Image, runtime: Any) -> torch.Tensor:
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
    """Contrast-normalise a CAM without letting a few outliers flatten it.

    Heatmaps often contain a handful of extreme pixels/patches. Min/max scaling
    makes every other region look almost invisible. Percentile scaling keeps the
    relative ordering but makes meaningful mid/high-activation areas readable.
    """
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

    # Slight gamma expansion makes medium-strength regions easier to see while
    # preserving stronger regions as warmer colours. This changes only the
    # display contrast, not the model prediction or the ordering of importance.
    array = np.power(array, 0.72)
    return torch.from_numpy(array.astype(np.float32))


def _overlay_png_data_url(image: Image.Image, cam: torch.Tensor) -> str:
    base = _resize_for_display(image)

    cam = cam.detach().float().cpu()[None, None, :, :]
    cam = F.interpolate(
        cam,
        size=(base.height, base.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam_array = cam.numpy()

    rgb = (_jet_colormap(cam_array) * 255).astype(np.uint8)
    heatmap = Image.fromarray(rgb, mode="RGB")

    # Strongly activated areas receive substantially more colour so the patch
    # regions are visually obvious. Low-activation regions still remain close
    # to the original photograph rather than being painted uniformly blue.
    visible_strength = np.power(np.clip(cam_array, 0.0, 1.0), 0.72)
    alpha = np.clip(visible_strength * OVERLAY_ALPHA, 0.0, OVERLAY_ALPHA)
    alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    heatmap.putalpha(alpha_image)

    composite = Image.alpha_composite(base.convert("RGBA"), heatmap).convert("RGB")

    buffer = io.BytesIO()
    composite.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _generate_gradcam_primary(
    image: Image.Image,
    runtime: Any,
    class_index: int,
) -> tuple[str, str]:
    """Return ``(PNG data URL, method name)`` for the predicted class."""
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
            logits = _forward_for_gradcam(image, runtime)
            if logits.ndim != 2 or class_index < 0 or class_index >= logits.shape[1]:
                raise RuntimeError("The predicted class index is invalid for Grad-CAM.")

            score = logits[0, class_index]
            score.backward()

        activations = captured.get("activations")
        gradients = captured.get("gradients")
        if activations is None or gradients is None:
            raise RuntimeError("The target layer did not expose activations and gradients.")

        activations = _as_spatial(activations.detach())
        gradients = _as_spatial(gradients.detach())

        if activations.shape != gradients.shape:
            raise RuntimeError(
                "Grad-CAM activation and gradient shapes did not match: "
                f"{tuple(activations.shape)} vs {tuple(gradients.shape)}"
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        weighted = (weights * activations).sum(dim=1)[0]
        cam = torch.relu(weighted)
        method = "Grad-CAM"

        minimum = cam.min()
        maximum = cam.max()
        if float((maximum - minimum).abs().item()) < 1e-12:
            # Some transformer/class combinations can make the signed
            # Grad-CAM map collapse after ReLU. Preserve a useful, honest
            # explanation by falling back to per-location gradient ×
            # activation magnitude and label it accordingly in the UI.
            cam = (gradients * activations).abs().mean(dim=1)[0]
            minimum = cam.min()
            maximum = cam.max()
            method = "Gradient × activation"

        if float((maximum - minimum).abs().item()) < 1e-12:
            raise RuntimeError("The model produced a flat attention map for this image.")

        cam = _robust_normalize_cam(cam)
        if runtime.key == "dinov2-base" and method == "Grad-CAM":
            method = "Transformer Grad-CAM"
        return _overlay_png_data_url(image, cam), method
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)



def _generate_input_gradient_data_url(
    image: Image.Image,
    runtime: Any,
    class_index: int,
) -> tuple[str, str]:
    """Fallback saliency map based on the gradient at the model input.

    This path is intentionally model-agnostic. It is used when a model's final
    transformer/convolution layer cannot provide a stable Grad-CAM activation
    map. The output is still class-specific because the gradient is taken from
    the selected class score.
    """
    model = runtime.model
    device = next(model.parameters()).device
    model.eval()
    model.zero_grad(set_to_none=True)

    if runtime.key != "dinov2-base" or runtime.model_type != "timm_dinov2":
        raise RuntimeError("This saliency implementation only supports DINOv2 Base.")

    with torch.enable_grad():
        input_tensor = (
            runtime.processor_or_transform(image)
            .unsqueeze(0)
            .to(device)
            .detach()
        )
        input_tensor.requires_grad_(True)
        logits = model(input_tensor)

        if logits.ndim != 2 or class_index < 0 or class_index >= logits.shape[1]:
            raise RuntimeError("The predicted class index is invalid for saliency mapping.")

        score = logits[0, class_index]
        score.backward()

        gradient = input_tensor.grad
        if gradient is None:
            raise RuntimeError("The model did not expose an input gradient.")

        # Gradient × input gives a class-specific contribution map. Taking the
        # absolute channel mean avoids cancelling positive and negative colour
        # contributions. A little average pooling makes the overlay easier to
        # read while preserving the model's broad focus regions.
        saliency = (gradient * input_tensor).abs().mean(dim=1, keepdim=True)
        if saliency.shape[-2] >= 3 and saliency.shape[-1] >= 3:
            saliency = F.avg_pool2d(saliency, kernel_size=5, stride=1, padding=2)
        cam = saliency[0, 0]

        minimum = cam.min()
        maximum = cam.max()
        if float((maximum - minimum).abs().item()) < 1e-12:
            cam = gradient.abs().mean(dim=1)[0]
            minimum = cam.min()
            maximum = cam.max()

        if float((maximum - minimum).abs().item()) < 1e-12:
            raise RuntimeError("The model produced a flat saliency map for this image.")

        cam = _robust_normalize_cam(cam)
        return _overlay_png_data_url(image, cam), "Input gradient × image"


def generate_gradcam_data_url(
    image: Image.Image,
    runtime: Any,
    class_index: int,
) -> tuple[str, str]:
    """Return a class-specific heatmap, with a robust saliency fallback.

    Grad-CAM remains the preferred method. If the selected model/layer cannot
    expose a usable activation-gradient map, the function automatically falls
    back to input-gradient saliency rather than dropping the heatmap entirely.
    """
    primary_error_text = "Unknown Grad-CAM error"
    try:
        return _generate_gradcam_primary(image, runtime, class_index)
    except Exception as error:
        # Keep only the message. Retaining the exception object can also retain
        # its traceback and large model tensors while the fallback runs.
        primary_error_text = str(error)

    try:
        return _generate_input_gradient_data_url(image, runtime, class_index)
    except Exception as fallback_error:
        raise RuntimeError(
            "Grad-CAM and the fallback saliency method both failed. "
            f"Grad-CAM: {primary_error_text}. Fallback: {fallback_error}."
        ) from fallback_error
