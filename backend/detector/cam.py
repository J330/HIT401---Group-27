"""Model-aware Grad-CAM for the four binary leaf classifiers.

Grad-CAM needs a spatial activation AND gradients of the *class logit* with
respect to it. Hugging Face classifiers return ImageClassifierOutput rather
than a logits tensor, while Swin/ViT expose spatial features as tokens. This
implementation operates on the logits explicitly and never changes the
predicted class/confidence.

For a finer view, its low-resolution CAM is guided by input-gradient
sensitivity for the same class logit. This is class-attribution visualization,
not a lesion detector; spots must come from actual gradients.

It intentionally does not run on the DINOv2 open-set gate; it explains only
the selected leaf-health classifier after the gate accepts the image.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from torch import nn


def _get_layer(model: nn.Module, model_key: str) -> tuple[nn.Module, bool]:
    """Return an activation module and whether it emits [B, tokens, channels]."""
    if model_key == "efficientnetv2s":
        layer = getattr(model, "conv_head", None)
        if layer is None:
            # Works for timm variants with no explicit convolution head.
            layer = next((m for m in reversed(list(model.modules()))
                          if isinstance(m, nn.Conv2d)), None)
        if layer is None:
            raise RuntimeError("No convolutional Grad-CAM layer in EfficientNetV2.")
        return layer, False

    if model_key == "convnextv2":
        # Target the penultimate stage (usually 14x14) for more localisation
        # than the final 7x7 stage. Its activations still contribute through
        # the last stage to the target class logit. For smaller/toy models,
        # use the final stage as before.
        stages = model.convnextv2.encoder.stages
        stage = stages[-2] if len(stages) >= 2 else stages[-1]
        return stage.layers[-1], False

    if model_key == "swin":
        # Swin's final stage has 7x7 spatial tokens at 224px. Capture before
        # the final attention block, which mixes these tokens into logits.
        stage = model.swin.encoder.layers[-1]
        block = stage.blocks[-1]
        return block.layernorm_before, True

    if model_key == "vit":
        # ViT classification reads the CLS token. After the last attention
        # operation the patch tokens no longer affect the CLS logit, so hooking
        # the LAST LAYER OUTPUT produces a blank map. Hook its input norm.
        return model.vit.encoder.layer[-1].layernorm_before, True

    raise ValueError(f"Unsupported classifier for Grad-CAM: {model_key}")


def _feature_map(features: torch.Tensor, token_features: bool) -> torch.Tensor:
    if not token_features:
        if features.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] features, got {tuple(features.shape)}")
        return features

    if features.ndim != 3:
        raise ValueError(f"Expected [B,tokens,channels] features, got {tuple(features.shape)}")
    n = features.shape[1]
    edge = math.isqrt(n)
    if edge * edge != n:
        edge = math.isqrt(n - 1)
        if edge * edge != n - 1:
            raise ValueError(f"Cannot infer spatial grid from {n} tokens")
        features = features[:, 1:, :]  # ViT's CLS token
    return features.transpose(1, 2).reshape(features.shape[0], features.shape[2], edge, edge)


def generate_gradcam(model: nn.Module, model_key: str, pixels: torch.Tensor,
                     target_class: int) -> np.ndarray:
    """Return a class-specific, 2D, 0..1 attention map (not a fake overlay).

    Using torch.autograd.grad avoids backward hooks on Hugging Face layers and
    works with PyTorch/timm/transformers version differences. Input gradients
    are enabled even when most pretrained backbone weights are frozen.
    """
    layer, tokens = _get_layer(model, model_key)
    saved: dict[str, torch.Tensor] = {}

    def capture(_module, _args, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Grad-CAM layer did not return a Tensor")
        saved["activations"] = output

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            # With frozen layers, activations have no gradient unless the
            # *input* also requires grad. Do not change trained weights.
            pixels_for_cam = pixels.detach().requires_grad_(True)
            if model_key == "efficientnetv2s":
                output = model(pixels_for_cam)
            else:
                output = model(pixel_values=pixels_for_cam)
            logits = output.logits if hasattr(output, "logits") else output
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise RuntimeError("Classifier must output logits shaped [B, classes]")
            if not (0 <= target_class < logits.shape[-1]):
                raise ValueError(f"Invalid Grad-CAM class index: {target_class}")
            activations = saved.get("activations")
            if activations is None:
                raise RuntimeError("Chosen Grad-CAM layer was not executed")
            gradients, input_gradients = torch.autograd.grad(
                logits[:, target_class].sum(), (activations, pixels_for_cam),
                retain_graph=False, create_graph=False,
            )

        a = _feature_map(activations.detach(), tokens)
        g = _feature_map(gradients.detach(), tokens)
        weights = g.mean(dim=(-2, -1), keepdim=True)
        heat = torch.relu((weights * a).sum(dim=1))[0]
        heat = heat - heat.min()
        maximum = heat.max()
        if not torch.isfinite(maximum):
            raise RuntimeError("Grad-CAM generated non-finite values")
        if maximum > 0:
            heat = heat / maximum
        # Finer *model-derived* detail: the original Grad-CAM is a coarse,
        # class-specific prior. Within those regions, input Gradient × Input
        # reveals finer pixels that influence the SAME class logit. This does
        # not invent lesion-shaped masks from the photograph's colour/edges.
        # It is a guided attribution view, not ordinary Grad-CAM or segmentation.
        size = tuple(pixels_for_cam.shape[-2:])
        coarse = torch.nn.functional.interpolate(
            heat[None, None], size=size, mode="bilinear", align_corners=False
        )[0, 0]
        sensitivity = (input_gradients.detach() * pixels_for_cam.detach()).abs()
        sensitivity = sensitivity.mean(dim=1, keepdim=True)
        # Smooth single-pixel gradient noise; keep true small response areas.
        sensitivity = torch.nn.functional.avg_pool2d(sensitivity, 5, 1, 2)[0, 0]
        if not torch.isfinite(sensitivity).all():
            return coarse.clamp(0, 1).cpu().numpy().astype(np.float32)
        scale = torch.quantile(sensitivity.flatten(), 0.98)
        if float(scale) <= 1e-12 or float(coarse.max()) <= 1e-12:
            return coarse.clamp(0, 1).cpu().numpy().astype(np.float32)
        sensitivity = (sensitivity / scale).clamp(0, 1)
        fine = coarse.pow(0.7) * sensitivity.pow(1.35)
        # Robust scaling prevents a single noisy pixel flattening the view.
        peak = torch.quantile(fine.flatten(), 0.995)
        if float(peak) > 1e-12:
            fine = fine / peak
        return fine.clamp(0, 1).cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
