"""Run with: python -m unittest discover -s tests -p 'test_cam_smoke.py' -v
Synthetic network tests; trained checkpoints are not distributed with this zip.
"""
import importlib.util
import pathlib
import types
import unittest

import numpy as np
import torch
from torch import nn

cam_file = pathlib.Path(__file__).resolve().parents[1] / 'detector' / 'cam.py'
spec = importlib.util.spec_from_file_location('cam_for_tests', cam_file)
cam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cam)


class ToyConvNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2 = nn.Module()
        self.convnextv2.encoder = nn.Module()
        self.convnextv2.encoder.stages = nn.ModuleList([nn.Module()])
        stage = self.convnextv2.encoder.stages[0]
        stage.layers = nn.ModuleList([nn.Conv2d(3, 8, 3, padding=1)])
        self.classifier = nn.Linear(8, 2)
        nn.init.constant_(self.classifier.weight, .5)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values):
        layer = self.convnextv2.encoder.stages[-1].layers[-1]
        x = layer(pixel_values)
        return types.SimpleNamespace(logits=self.classifier(x.mean(dim=(2, 3))))


class ToyEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_head = nn.Conv2d(3, 8, 3, padding=1)
        self.classifier = nn.Linear(8, 2)

    def forward(self, pixels):
        return self.classifier(self.conv_head(pixels).mean(dim=(2, 3)))


class ToyTokenNet(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit_mode = vit
        self.to_tokens = nn.Conv2d(3, 8, 2, stride=2)
        self.classifier = nn.Linear(8, 2)
        self.norm = nn.LayerNorm(8)
        if vit:
            self.vit = nn.Module()
            self.vit.encoder = nn.Module()
            self.vit.encoder.layer = nn.ModuleList([nn.Module()])
            self.vit.encoder.layer[-1].layernorm_before = self.norm
        else:
            self.swin = nn.Module()
            self.swin.encoder = nn.Module()
            self.swin.encoder.layers = nn.ModuleList([nn.Module()])
            self.swin.encoder.layers[-1].blocks = nn.ModuleList([nn.Module()])
            self.swin.encoder.layers[-1].blocks[-1].layernorm_before = self.norm

    def forward(self, pixel_values):
        x = self.to_tokens(pixel_values).flatten(2).transpose(1, 2)
        if self.vit_mode:
            x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # pretend CLS token
        x = self.norm(x)
        return types.SimpleNamespace(logits=self.classifier(x.mean(dim=1)))


class CamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pixels = torch.randn(1, 3, 8, 8)

    def check(self, model, key, expected_shape):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)  # even a fully frozen backbone should CAM
        actual = cam.generate_gradcam(model, key, self.pixels, 0)
        self.assertEqual(actual.shape, expected_shape)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        # Hooks cleaned up after use
        target, _ = cam._get_layer(model, key)
        self.assertEqual(len(target._forward_hooks), 0)

    def test_convnext_hf_logits_and_spatial(self):
        self.check(ToyConvNext(), 'convnextv2', (8, 8))

    def test_convnext_with_two_stages_uses_higher_resolution_features(self):
        class TwoStageConvNext(nn.Module):
            def __init__(self):
                super().__init__()
                self.convnextv2 = nn.Module()
                self.convnextv2.encoder = nn.Module()
                stage1 = nn.Module()
                stage1.layers = nn.ModuleList([nn.Conv2d(3, 8, 3, padding=1)])
                stage2 = nn.Module()
                stage2.layers = nn.ModuleList([nn.Conv2d(8, 8, 3, stride=2, padding=1)])
                self.convnextv2.encoder.stages = nn.ModuleList([stage1, stage2])
                self.classifier = nn.Linear(8, 2)

            def forward(self, pixel_values):
                stages = self.convnextv2.encoder.stages
                x = stages[0].layers[-1](pixel_values)
                x = stages[1].layers[-1](x)
                return types.SimpleNamespace(logits=self.classifier(x.mean((2, 3))))

        self.check(TwoStageConvNext(), 'convnextv2', (8, 8))

    def test_effnet_tensor_logits(self):
        self.check(ToyEffNet(), 'efficientnetv2s', (8, 8))

    def test_swin_token_grid(self):
        self.check(ToyTokenNet(vit=False), 'swin', (8, 8))

    def test_vit_cls_token_removed(self):
        self.check(ToyTokenNet(vit=True), 'vit', (8, 8))

    def test_error_unknown_model(self):
        with self.assertRaises(ValueError):
            cam.generate_gradcam(ToyEffNet(), 'unknown', self.pixels, 0)

    def test_non_square_tokens_rejected(self):
        with self.assertRaises(ValueError):
            cam._feature_map(torch.zeros((1, 18, 8)), True)


if __name__ == '__main__':
    unittest.main()
