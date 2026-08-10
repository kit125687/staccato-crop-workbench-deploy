import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

from backend.outpaint import (
    AI_CONFIG,
    OutpaintError,
    _foreground_validation_mask,
    _gemini_outpaint,
    _request_work_size,
    _validate_protected_pixels,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class GeminiOutpaintTests(unittest.TestCase):
    def test_gemini_portrait_request_keeps_three_by_four_geometry(self):
        self.assertEqual(_request_work_size("gemini", (3000, 4000)), (1536, 2048))
        self.assertEqual(_request_work_size("openai", (3000, 4000)), (1024, 1536))

    def test_subject_pixel_guard_accepts_unchanged_source(self):
        source = Image.new("RGBA", (16, 16), (80, 90, 100, 255))
        mask = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        _validate_protected_pixels(source.convert("RGB"), source, mask)

    def test_subject_pixel_guard_allows_tiny_jpeg_noise(self):
        source = Image.new("RGBA", (16, 16), (80, 90, 100, 255))
        mask = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        _validate_protected_pixels(Image.new("RGB", (16, 16), (82, 91, 99)), source, mask)

    def test_foreground_mask_excludes_background_and_keeps_product(self):
        source = Image.new("RGB", (100, 100), "white")
        for y in range(30, 80):
            for x in range(25, 75):
                source.putpixel((x, y), (120, 70, 30))
        mask = _foreground_validation_mask(source, (10, 20), (140, 150), None)
        alpha = mask.getchannel("A")
        self.assertEqual(alpha.getpixel((15, 25)), 0)
        self.assertEqual(alpha.getpixel((60, 70)), 255)

    def test_subject_pixel_guard_rejects_small_global_colour_shift(self):
        source = Image.new("RGBA", (16, 16), (80, 90, 100, 255))
        mask = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        with self.assertRaisesRegex(OutpaintError, "主体颜色/细节保护校验未通过"):
            _validate_protected_pixels(Image.new("RGB", (16, 16), (87, 97, 107)), source, mask)

    def test_subject_pixel_guard_rejects_recolored_or_reframed_source(self):
        source = Image.new("RGBA", (16, 16), (80, 90, 100, 255))
        mask = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        with self.assertRaisesRegex(OutpaintError, "主体颜色/细节保护校验未通过"):
            _validate_protected_pixels(Image.new("RGB", (16, 16), "white"), source, mask)

    def test_requests_jpeg_response_while_sending_png_input(self):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(output, "JPEG")
        captured = {}

        def fake_urlopen(request, timeout=0):
            captured.update(json.loads(request.data))
            return _Response({"output_image": {"data": base64.b64encode(output.getvalue()).decode()}})

        previous = AI_CONFIG.copy()
        AI_CONFIG.update({"provider": "gemini", "base_url": "https://example.test/v1beta", "model": "gemini-3.1-flash-lite-image", "api_key": "test"})
        try:
            with patch("backend.outpaint.urllib.request.urlopen", side_effect=fake_urlopen):
                result = _gemini_outpaint(Image.new("RGBA", (32, 48), (0, 0, 0, 0)), (300, 400), "expand")
        finally:
            AI_CONFIG.clear()
            AI_CONFIG.update(previous)

        self.assertEqual(result.size, (300, 400))
        self.assertEqual(captured["input"][1]["mime_type"], "image/png")
        self.assertEqual(captured["response_format"]["mime_type"], "image/jpeg")
        self.assertEqual(captured["response_format"]["image_size"], "1K")

    def test_nano_banana_pro_requests_4k_jpeg(self):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(output, "JPEG")
        captured = {}

        def fake_urlopen(request, timeout=0):
            captured.update(json.loads(request.data))
            return _Response({"steps": [
                {"type": "user_input", "content": [{"type": "text", "text": "expand"}]},
                {"type": "model_output", "content": [{"type": "text", "text": "done"}, {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(output.getvalue()).decode()}]},
            ]})

        previous = AI_CONFIG.copy()
        AI_CONFIG.update({"provider": "gemini", "base_url": "https://example.test/v1beta", "model": "gemini-3-pro-image", "api_key": "test"})
        try:
            with patch("backend.outpaint.urllib.request.urlopen", side_effect=fake_urlopen):
                _gemini_outpaint(Image.new("RGBA", (32, 48), (0, 0, 0, 0)), (3000, 4000), "expand")
        finally:
            AI_CONFIG.clear()
            AI_CONFIG.update(previous)

        self.assertEqual(captured["model"], "gemini-3-pro-image")
        self.assertEqual(captured["response_format"]["mime_type"], "image/jpeg")
        self.assertEqual(captured["response_format"]["aspect_ratio"], "3:4")
        self.assertEqual(captured["response_format"]["image_size"], "4K")


if __name__ == "__main__":
    unittest.main()
