"""Sanity checks for generated heatmap pixels (not an AI model evaluation)."""
import ast
import base64
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'detector' / 'heatmap.py'
spec = importlib.util.spec_from_file_location('heatmap_for_tests', MODULE)
heatmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heatmap)


class HeatmapStyleTests(unittest.TestCase):
    def test_annotations_are_compatible_with_older_python(self):
        # Python <3.10 cannot evaluate ``Image.Image | None`` at import time.
        source = MODULE.read_text(encoding="utf-8")
        syntax = ast.parse(source)
        # Enforce stable, evaluated Optional/ Tuple annotations, rather than
        # runtime evaluation of PEP 604 unions on Python 3.9.
        for node in ast.walk(syntax):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations = [arg.annotation for arg in node.args.args]
                annotations.append(node.returns)
                for annotation in annotations:
                    if annotation is not None:
                        self.assertFalse(
                            any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
                                for child in ast.walk(annotation)),
                            f"Python 3.9-incompatible annotation in {node.name}",
                        )

    def test_uniform_map_does_not_invent_hotspots(self):
        output = heatmap._enhance_cam(np.full((8, 8), 0.4, dtype=np.float32))
        self.assertTrue(np.all(output == 0.0))

    def test_coloured_overlay_shows_blue_low_influence_and_red_hotspots(self):
        image = Image.new('RGB', (64, 64), (100, 100, 100))
        raw = np.zeros((64, 64), np.float32)
        raw[20:44, 20:44] = 1
        gray_url, colour_url = heatmap.make_heatmap_urls(image, raw)
        self.assertTrue(gray_url.startswith('data:image/png;base64,'))
        out = Image.open(io.BytesIO(base64.b64decode(colour_url.split(',', 1)[1]))).convert('RGB')
        # The entire image is colour-coded, even where attribution is zero.
        low_red, low_green, low_blue = out.getpixel((0, 0))
        self.assertGreater(low_blue, low_red)
        self.assertGreater(low_blue, low_green)
        self.assertNotEqual(out.getpixel((0, 0)), (100, 100, 100))
        red, green, blue = out.getpixel((32, 32))
        self.assertGreater(red, green)
        self.assertGreater(red, blue)

    def test_high_attention_expansion_is_local_and_keeps_distant_blue(self):
        raw = np.zeros((120, 120), dtype=np.float32)
        raw[58:62, 58:62] = 1.0
        enhanced = heatmap._enhance_cam(raw)
        expanded = heatmap._expand_hotspots(enhanced)
        self.assertGreater(
            np.count_nonzero(expanded >= 0.91),
            np.count_nonzero(enhanced >= 0.91),
        )
        self.assertEqual(float(expanded[0, 0]), 0.0)
        self.assertEqual(float(expanded[15, 15]), 0.0)
        self.assertEqual(float(expanded[60, 60]), 1.0)

    def test_expansion_does_not_create_hotspots_without_strong_seed(self):
        raw = np.zeros((60, 60), dtype=np.float32)
        raw[20:40, 20:40] = 0.5
        expanded = heatmap._expand_hotspots(raw)
        self.assertTrue(np.array_equal(expanded, raw))

    def test_mask_limits_red_hotspots_and_preserves_grey_background(self):
        original = Image.new('RGB', (120, 100), (128, 128, 128))
        mask = Image.new('L', (120, 100), 0)
        for y in range(24, 76):
            for x in range(24, 96):
                mask.putpixel((x, y), 255)
        raw = np.zeros((100, 120), np.float32)
        raw[45:55, 54:64] = 1.0      # hot leaf region
        raw[4:19, 4:19] = 1.0        # hot background region: must be invisible
        gray_url, overlay_url = heatmap.make_heatmap_urls(original, raw, foreground_mask=mask)
        heat = Image.open(io.BytesIO(base64.b64decode(gray_url.split(',', 1)[1]))).convert('RGBA')
        overlay = Image.open(io.BytesIO(base64.b64decode(overlay_url.split(',', 1)[1]))).convert('RGB')
        self.assertEqual(heat.size, original.size)
        self.assertEqual(heat.getpixel((10, 10))[3], 0)
        self.assertEqual(heat.getpixel((108, 95))[3], 0)
        self.assertEqual(overlay.getpixel((10, 10)), (128, 128, 128))
        self.assertEqual(overlay.getpixel((108, 95)), (128, 128, 128))
        self.assertEqual(overlay.getpixel((10, 13)), (128, 128, 128))
        self.assertGreater(heat.getpixel((58, 50))[3], 200)
        r, g, b = overlay.getpixel((58, 50))
        self.assertGreater(r, g)
        self.assertGreater(r, b)
        self.assertGreater(overlay.getpixel((30, 50))[2], overlay.getpixel((30, 50))[0])

    def test_feathered_edges_and_mask_alignment_after_resize(self):
        original = Image.new('RGB', (1200, 600), (128, 128, 128))
        mask = Image.new('L', original.size, 0)
        for y in range(100, 500):
            for x in range(200, 1000):
                mask.putpixel((x, y), 255)
        raw = np.zeros((20, 40), dtype=np.float32)
        raw[5:15, 10:30] = 1
        url, overlay_url = heatmap.make_heatmap_urls(original, raw, foreground_mask=mask)
        heat = Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGBA')
        overlay = Image.open(io.BytesIO(base64.b64decode(overlay_url.split(',', 1)[1]))).convert('RGB')
        self.assertEqual(heat.size, (900, 450))
        self.assertEqual(heat.getpixel((0, 0))[3], 0)
        self.assertEqual(heat.getpixel((450, 225))[3], 255)
        self.assertEqual(overlay.getpixel((0, 0)), (128, 128, 128))
        # Feather transitions should be between zero and full opacity at edges.
        alphas = [heat.getpixel((x, 225))[3] for x in range(143, 158)]
        self.assertTrue(any(0 < a < 255 for a in alphas))

    def test_mismatched_mask_is_rejected_not_silently_shifted(self):
        with self.assertRaisesRegex(ValueError, 'matching sizes'):
            heatmap.make_heatmap_urls(
                Image.new('RGB', (40, 40)), np.ones((10, 10), np.float32),
                foreground_mask=Image.new('L', (45, 40), 255),
            )

    def test_missing_mask_keeps_previous_full_frame_fallback(self):
        image = Image.new('RGB', (32, 32), (100, 100, 100))
        raw = np.zeros((32, 32), np.float32)
        raw[14:18, 14:18] = 1
        gray_url, _ = heatmap.make_heatmap_urls(image, raw)
        heat = Image.open(io.BytesIO(base64.b64decode(gray_url.split(',', 1)[1]))).convert('RGBA')
        self.assertEqual(heat.getchannel('A').getextrema(), (255, 255))

    def test_uniform_attention_renders_uniform_blue_without_fake_hotspots(self):
        image = Image.new('RGB', (32, 32), (120, 120, 120))
        raw = np.ones((32, 32), dtype=np.float32) * 0.55
        gray_url, overlay_url = heatmap.make_heatmap_urls(image, raw)
        grayscale = Image.open(io.BytesIO(base64.b64decode(gray_url.split(',', 1)[1]))).convert('RGBA')
        overlay = Image.open(io.BytesIO(base64.b64decode(overlay_url.split(',', 1)[1]))).convert('RGB')
        self.assertEqual(grayscale.getchannel("R").getextrema(), (0, 0))
        self.assertEqual(grayscale.getchannel("A").getextrema(), (255, 255))
        self.assertEqual(overlay.getpixel((0, 0)), overlay.getpixel((16, 16)))
        red, green, blue = overlay.getpixel((16, 16))
        self.assertGreater(blue, red)


if __name__ == '__main__':
    unittest.main()
