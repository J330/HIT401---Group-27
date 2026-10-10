"""No pretrained downloads needed; mocks validate the scan wiring and mask fallback."""
import ast
import importlib.util
import io
import logging
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest import TestCase, mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("background_filter_test", ROOT / "detector" / "background_filter.py")
bg = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = bg
spec.loader.exec_module(bg)


def load_predict_function(namespace):
    """Load just the pure orchestration function, avoiding large model init."""
    source = ast.parse((ROOT / "detector" / "inference.py").read_text())
    fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "predict_image")
    ast.fix_missing_locations(fn)
    code = compile(ast.Module(body=[fn], type_ignores=[]), "<predict_image>", "exec")
    exec(code, namespace)
    return namespace["predict_image"]


class ForegroundTests(TestCase):
    def test_mask_crops_foreground_on_neutral_grey(self):
        original = Image.new("RGB", (100, 100), (70, 20, 20))
        rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        leaf = Image.new("RGBA", (50, 40), (30, 150, 35, 255))
        rgba.paste(leaf, (20, 25))
        with mock.patch.object(bg, "_get_rembg_session", return_value=object()), \
             mock.patch.object(bg, "_to_rgba_removed", return_value=rgba):
            out = bg.make_background_filtered_view(original)
        self.assertTrue(out.applied)
        self.assertEqual(out.method, "rembg:u2netp")
        self.assertLess(out.image.width, original.width)
        self.assertEqual(out.image.getpixel((out.image.width//2, out.image.height//2)), (30, 150, 35))
        self.assertEqual(out.image.getpixel((0,0)), (128, 128, 128))
        self.assertIsNotNone(out.foreground_mask)
        self.assertEqual(out.foreground_mask.size, out.image.size)
        self.assertEqual(out.foreground_mask.getpixel((0, 0)), 0)
        self.assertEqual(out.foreground_mask.getpixel((out.image.width//2, out.image.height//2)), 255)

    def test_failed_background_removal_preserves_original(self):
        source = Image.new("RGB", (80, 80), (30, 60, 90))
        with mock.patch.object(bg, "_get_rembg_session", side_effect=RuntimeError("offline")):
            out = bg.make_background_filtered_view(source)
        self.assertFalse(out.applied)
        self.assertEqual(out.image.tobytes(), source.tobytes())
        self.assertIn("offline", out.reason)
        self.assertIsNone(out.foreground_mask)

    def test_rejects_empty_mask(self):
        source = Image.new("RGB", (100, 100), (10, 20, 30))
        with mock.patch.object(bg, "_get_rembg_session", return_value=object()), \
             mock.patch.object(bg, "_to_rgba_removed", return_value=Image.new("RGBA", (100, 100), (0, 0, 0, 0))):
            out = bg.make_background_filtered_view(source)
        self.assertFalse(out.applied)
        self.assertEqual(out.image.size, source.size)


class IntegrationWiringTests(TestCase):
    def setUp(self):
        self.photo = Image.new("RGB", (80, 80), (10, 20, 30))
        self.processed = Image.new("RGB", (35, 60), (80, 90, 100))
        self.mask = Image.new("L", self.processed.size, 255)
        data = io.BytesIO()
        self.photo.save(data, "PNG")
        self.input_file = io.BytesIO(data.getvalue())
        self.classified_images = []
        self.prepared_images = []
        self.explained = []
        self.original_tensor = object()
        self.processed_tensor = object()
        self.gated_images = []
        self.filter_calls = []
        self.call_order = []

        def gate(image):
            self.gated_images.append(image.copy())
            return True, "healthy", 0.12

        def classify(image):
            self.call_order.append("classify-original")
            self.classified_images.append(image.copy())
            return "healthy", 0.82, self.original_tensor, 1

        def prepare(image):
            self.prepared_images.append(image)
            return self.processed_tensor

        def explain(pixels, selected_class):
            self.explained.append((pixels, selected_class))
            return [[0.0, 1.0]]

        def filter_background(image):
            self.call_order.append("filter-for-heatmap")
            self.filter_calls.append(image.copy())
            return bg.BackgroundFilterResult(self.processed, True, "rembg:u2netp", foreground_mask=self.mask)

        self.n = {
            "Image": Image, "ImageOps": __import__("PIL.ImageOps", fromlist=["exif_transpose"]),
            "_gate_check": gate,
            "_classify": classify,
            "_prepare_classifier_input": prepare,
            "_explain": explain,
            "_CLASSIFIER_LOCK": RLock(),
            "logger": logging.getLogger("unit-tests"),
            "settings": SimpleNamespace(DEBUG=True),
            "ACTIVE_MODEL_KEY": "vit", "GATE_THRESHOLD": 0.30,
            "make_background_filtered_view": filter_background,
        }
        self.predict = load_predict_function(self.n)

    def test_original_photo_drives_prediction_before_isolation(self):
        result = self.predict(self.input_file)
        self.assertEqual(self.gated_images[0].tobytes(), self.photo.tobytes())
        self.assertEqual(self.classified_images[0].tobytes(), self.photo.tobytes())
        self.assertEqual(result["label"], "healthy")
        self.assertEqual(result["confidence"], 0.82)
        self.assertEqual(len(self.classified_images), 1)
        self.assertEqual(len(self.filter_calls), 1)
        self.assertEqual(self.call_order, ["classify-original", "filter-for-heatmap"])

    def test_automatic_isolation_uses_processed_view_for_cam_only(self):
        result = self.predict(self.input_file)
        self.assertIs(self.prepared_images[0], self.processed)
        self.assertEqual(self.explained, [(self.processed_tensor, 1)])
        self.assertIs(result["heatmap_source_image"], self.processed)
        self.assertTrue(result["background_removed_applied"])
        self.assertIs(result["heatmap_foreground_mask"], self.mask)
        self.assertNotIn("background_removal_requested", result)

    def test_isolation_failure_keeps_original_prediction_and_cam(self):
        self.n["make_background_filtered_view"] = lambda i: bg.BackgroundFilterResult(i, False, "original-only fallback", "offline")
        self.predict = load_predict_function(self.n)
        result = self.predict(self.input_file)
        self.assertEqual(self.classified_images[0].tobytes(), self.photo.tobytes())
        self.assertFalse(result["background_removed_applied"])
        self.assertEqual(result["background_removal_reason"], "offline")
        self.assertEqual(self.prepared_images, [])
        self.assertEqual(self.explained, [(self.original_tensor, 1)])
        self.assertEqual(result["confidence"], 0.82)
        self.assertIsNone(result["heatmap_foreground_mask"])

    def test_rejection_skips_isolation_and_cam(self):
        self.n["_gate_check"] = lambda i: (False, "unknown", 0.60)
        self.n["make_background_filtered_view"] = lambda i: self.fail("must not isolate a rejected image")
        self.predict = load_predict_function(self.n)
        result = self.predict(self.input_file)
        self.assertIsNone(result["heatmap_array"])
        self.assertIsNone(result["heatmap_source_image"])
        self.assertIsNone(result["heatmap_foreground_mask"])
        self.assertEqual(self.classified_images, [])
        self.assertEqual(self.explained, [])

    def test_cam_failure_keeps_disease_prediction(self):
        self.n["_explain"] = lambda pixels, cl: (_ for _ in ()).throw(RuntimeError("CAM failure"))
        self.predict = load_predict_function(self.n)
        with self.assertLogs("unit-tests", level="ERROR"):
            result = self.predict(self.input_file)
        self.assertEqual(result["label"], "healthy")
        self.assertEqual(result["confidence"], 0.82)
        self.assertIsNone(result["heatmap_array"])
        self.assertIn("CAM failure", result["heatmap_error"])

    def test_upload_page_has_no_background_removal_toggle(self):
        template = (ROOT / "templates" / "upload.html").read_text()
        upload_js = (ROOT / "static" / "js" / "upload.js").read_text()
        api_js = (ROOT / "static" / "js" / "api.js").read_text()
        self.assertNotIn('id="remove-background"', template)
        self.assertNotIn('removeBackground', upload_js)
        self.assertNotIn('formData.append("remove_background"', api_js)


class FrontendMaskWiringTests(TestCase):
    def test_canvas_reads_png_alpha_and_masks_colour(self):
        js = (ROOT / "static" / "js" / "result.js").read_text()
        self.assertIn("camMask[i] = data[i * 4 + 3] / 255", js)
        self.assertIn("* camMask[i]", js)
        self.assertIn("Background excluded", js)

    def test_pdf_uses_masked_server_overlay_and_leaf_canvas(self):
        js = (ROOT / "static" / "js" / "result.js").read_text()
        self.assertIn('showPrintHeatmap(result.heatmap_overlay_url)', js)
        self.assertIn('ctx.drawImage(canvas, 0, 0, w, h)', js)
