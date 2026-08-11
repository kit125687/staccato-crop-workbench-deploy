import tempfile
import unittest
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.core import Box, ImageItem, OUTPUT_SIZES, _jpeg_under_limit, _paste_seamless, analyze_image, assign_barcodes, export_items, is_generated_output, render_output, safe_zone, scan_root
from unittest.mock import patch


def item(folder, image_type, index):
    return ImageItem(str(index), folder, f"/tmp/{index}.jpg", f"{index}.jpg", 3000, 4000, image_type, .9, "test", Box(200, 200, 1200, 1800), "solid", False)


class CoreTests(unittest.TestCase):
    def test_barcode_priority_and_shared_pool(self):
        items = [item("SKU", "静物", 1), item("SKU", "静物", 2), item("SKU", "静物", 3), item("SKU", "腿模", 4), item("SKU", "腿模", 5), item("SKU", "全身", 6)]
        assign_barcodes(items)
        self.assertEqual([i.barcodes for i in items], [["43"], ["44"], ["42"], ["45", "46"], ["48"], ["50", "51"]])

    def test_exact_safe_zones(self):
        self.assertEqual(safe_zone("静物", (3000, 3000)), Box(668, 640, 1664, 1665))
        self.assertEqual(safe_zone("腿模", (3000, 4000)), Box(160, 768, 2680, 2400))
        self.assertNotEqual(safe_zone("全身", (3000, 4000)), Box(160, 768, 2680, 2400))

    def test_custom_output_size_is_used_and_safe_zone_scales(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "custom.jpg"
            Image.new("RGB", (3000, 4000), "white").save(path)
            source = ImageItem("custom", "SKU", str(path), path.name, 3000, 4000, "静物", .9, "test", Box(200, 200, 1200, 1800), "solid", False)
            source.output_sizes["43"] = [1200, 1600]
            output, _ = render_output(source, "43", allow_complex_fallback=True)
        self.assertEqual(output.size, (1200, 1600))
        self.assertEqual(safe_zone("静物", (1200, 1600)), Box(64, 307, 1072, 960))

    def test_jpeg_encoder_enforces_two_megabyte_limit(self):
        pixels = np.random.default_rng(7).integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
        payload, quality = _jpeg_under_limit(Image.fromarray(pixels))
        self.assertLessEqual(len(payload), 2_000_000)
        self.assertGreaterEqual(quality, 35)

    def test_custom_output_name_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.jpg"
            Image.new("RGB", (800, 800), "white").save(path)
            source = ImageItem("named", "SKU", str(path), path.name, 800, 800, "静物", .9, "test", Box(0, 0, 800, 800), "solid", False)
            source.barcodes = ["43"]
            source.output_names["43"] = "自定义主图.jpg"
            exported, failed = export_items([source])
            self.assertFalse(failed)
            self.assertEqual(Path(exported[0]["path"]).name, "自定义主图.jpg")
            self.assertLessEqual(exported[0]["bytes"], 2_000_000)

    def test_generated_output_is_excluded(self):
        self.assertTrue(is_generated_output(Path("ABC_43.jpg"), "ABC"))
        self.assertTrue(is_generated_output(Path("条码_50.jpg"), "ABC"))
        self.assertFalse(is_generated_output(Path("original_43.jpg"), "ABC"))

    def test_folder_scan_and_static_classification(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "SKU001"; folder.mkdir()
            image = Image.new("RGB", (900, 1200), "white")
            ImageDraw.Draw(image).ellipse((180, 650, 720, 900), fill=(25, 25, 25))
            image.save(folder / "shoe.jpg")
            image.save(folder / "SKU001_43.jpg")
            items, errors = scan_root(root)
            self.assertFalse(errors)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image_type, "静物")
            self.assertEqual(items[0].barcodes, ["43"])

    def test_edge_contact_is_not_manual_review(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "SKU002"; folder.mkdir()
            image = Image.new("RGB", (900, 1200), "white")
            ImageDraw.Draw(image).rectangle((0, 650, 650, 910), fill=(20, 20, 20))
            image.save(folder / "edge-shoe.jpg")
            items, errors = scan_root(root)
            self.assertFalse(errors)
            self.assertNotIn("主体贴近原图边缘", items[0].review_reasons)

    def test_local_pose_marks_lower_body_as_leg_model(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "leg.jpg"
            Image.new("RGB", (900, 1200), "white").save(path)
            pose = [
                {"name": "left_knee", "x": .4, "y": .45, "confidence": .9},
                {"name": "left_ankle", "x": .4, "y": .8, "confidence": .9},
                {"name": "left_foot", "x": .5, "y": .86, "confidence": .9},
            ]
            with patch("backend.core._vision_body_pose", return_value=[]), patch("backend.core._mediapipe_body_pose", return_value=pose):
                result = analyze_image(path, "SKU")
            self.assertEqual(result.image_type, "腿模")

    def test_local_pose_requires_upper_torso_for_full_body(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "full.jpg"
            Image.new("RGB", (900, 1200), "white").save(path)
            pose = [
                {"name": name, "x": .5, "y": y, "confidence": .9}
                for name, y in (("left_shoulder", .2), ("right_shoulder", .2), ("left_hip", .48), ("left_knee", .7), ("left_ankle", .9))
            ]
            with patch("backend.core._vision_body_pose", return_value=[]), patch("backend.core._mediapipe_body_pose", return_value=pose):
                result = analyze_image(path, "SKU")
            self.assertEqual(result.image_type, "全身")

    def test_clean_extension_feathers_rectangular_boundary(self):
        canvas = Image.new("RGB", (240, 240), (232, 234, 236))
        source = Image.new("RGB", (160, 160), (250, 250, 250))
        _paste_seamless(canvas, source, (40, 40))
        self.assertEqual(canvas.getpixel((40, 100)), (232, 234, 236))
        self.assertNotEqual(canvas.getpixel((55, 100)), (232, 234, 236))

    def test_product_detail_covers_canvas_without_outpainting(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "detail.jpg"
            Image.new("RGB", (900, 1200), (210, 180, 150)).save(path)
            detail = ImageItem("detail", "SKU", str(path), path.name, 900, 1200, "静物", .95, "产品细节", Box(0, 0, 900, 1200), "complex", False)
            output, method = render_output(detail, "42")
            self.assertEqual(output.size, (3000, 4000))
            self.assertEqual(method, "direct")

    def test_ai_failure_still_exports_positioned_unextended_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.jpg"
            Image.new("RGB", (1000, 1000), (220, 220, 220)).save(path)
            source = ImageItem("fallback", "SKU", str(path), path.name, 1000, 1000, "静物", .9, "test", Box(50, 50, 900, 900), "complex", False)
            source.barcodes = ["43"]
            with patch("backend.outpaint.gpt_outpaint", side_effect=RuntimeError("HTTP 400 detail")):
                exported, failed = export_items([source])
            self.assertFalse(failed)
            self.assertEqual(exported[0]["method"], "fallback_unextended")
            self.assertIn("HTTP 400 detail", exported[0]["warning"])
            with Image.open(Path(root) / "SKU_43.jpg") as generated:
                self.assertEqual(generated.size, (3000, 4000))

    def test_no_ai_mode_leaves_uncovered_canvas_white(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "no-ai.jpg"
            Image.new("RGB", (800, 800), (180, 120, 80)).save(path)
            source = ImageItem("no-ai", "SKU", str(path), path.name, 800, 800, "静物", .9, "test", Box(100, 100, 600, 600), "complex", False)
            with patch.dict(os.environ, {"NO_AI_MODE": "true"}):
                output, method = render_output(source, "43")
            self.assertEqual(method, "no_ai_blank")
            self.assertEqual(output.size, (3000, 4000))

    def test_blank_completion_mode_does_not_call_ai(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "blank-mode.jpg"
            Image.new("RGB", (800, 800), (180, 120, 80)).save(path)
            source = ImageItem("blank-mode", "SKU", str(path), path.name, 800, 800, "静物", .9, "test", Box(100, 100, 600, 600), "complex", False)
            with patch("backend.outpaint.gpt_outpaint", side_effect=AssertionError("AI must not be called")):
                output, method = render_output(source, "43", completion_mode="blank")
            self.assertEqual(method, "no_ai_blank")
            self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
