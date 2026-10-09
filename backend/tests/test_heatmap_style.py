"""Sanity checks for generated heatmap pixels (not an AI model evaluation)."""
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
    def test_uniform_map_does_not_invent_hotspots(self):
        output = heatmap._enhance_cam(np.full((8, 8), 0.4, dtype=np.float32))
        self.assertTrue(np.all(output == 0.0))

    def test_coloured_overlay_has_red_hotspots_and_transparent_background(self):
        image = Image.new('RGB', (64, 64), (100, 100, 100))
        raw = np.zeros((64, 64), np.float32)
        raw[20:44, 20:44] = 1
        gray_url, colour_url = heatmap.make_heatmap_urls(image, raw)
        self.assertTrue(gray_url.startswith('data:image/png;base64,'))
        out = Image.open(io.BytesIO(base64.b64decode(colour_url.split(',', 1)[1]))).convert('RGB')
        # The original backdrop stays visible outside significant attention.
        self.assertEqual(out.getpixel((0, 0)), (100, 100, 100))
        red, green, blue = out.getpixel((32, 32))
        self.assertGreater(red, green)
        self.assertGreater(red, blue)


if __name__ == '__main__':
    unittest.main()
