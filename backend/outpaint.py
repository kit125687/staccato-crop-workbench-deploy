from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageFilter


class OutpaintError(RuntimeError):
    pass


KEYCHAIN_SERVICE = "com.staccato.crop-workbench.api-key"
KEYCHAIN_CONFIG_SERVICE = "com.staccato.crop-workbench.ai-config"


def _keychain_key(provider: str) -> str:
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.run(
            ["security", "find-generic-password", "-s", f"{KEYCHAIN_SERVICE}.{provider}", "-w"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return ""


def _save_keychain_key(provider: str, api_key: str) -> None:
    if sys.platform != "darwin":
        return
    subprocess.run(
        ["security", "add-generic-password", "-U", "-s", f"{KEYCHAIN_SERVICE}.{provider}", "-a", os.getenv("USER", "local"), "-w", api_key],
        check=True, capture_output=True, text=True,
    )


def _keychain_config() -> dict:
    if sys.platform != "darwin":
        return {}
    try:
        value = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_CONFIG_SERVICE, "-w"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return json.loads(value)
    except Exception:
        return {}


def _save_keychain_config(config: dict) -> None:
    if sys.platform != "darwin":
        return
    subprocess.run(
        ["security", "add-generic-password", "-U", "-s", KEYCHAIN_CONFIG_SERVICE, "-a", os.getenv("USER", "local"), "-w", json.dumps(config)],
        check=True, capture_output=True, text=True,
    )


_SAVED_AI = _keychain_config()
AI_CONFIG = {
    "enabled": os.getenv("IMAGE_AI_ENABLED", "true").lower() != "false",
    "provider": os.getenv("IMAGE_AI_PROVIDER", _SAVED_AI.get("provider", "openai")),
    "base_url": os.getenv("OPENAI_BASE_URL", _SAVED_AI.get("base_url", "https://api.openai.com/v1")),
    "model": os.getenv("IMAGE_AI_MODEL", _SAVED_AI.get("model", "gpt-image-2")),
    "api_key": os.getenv("OPENAI_API_KEY", "") or _keychain_key(_SAVED_AI.get("provider", "openai")),
}


def _validate_gemini_key(api_key: str) -> None:
    request = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=20):
            return
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            detail = body.get("error", {}).get("message", f"HTTP {exc.code}")
        except Exception:
            detail = f"HTTP {exc.code} {exc.reason}"
        raise ValueError(f"Gemini Key 验证失败：{detail}") from exc


def configure_ai(enabled: bool, base_url: str, model: str, api_key: str = "", provider: str = "openai") -> dict:
    if provider not in {"openai", "gemini"}:
        raise ValueError("不支持的 AI 服务类型")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("AI 接口地址必须是有效的 http(s) 链接")
    clean_key = api_key.strip()
    if not clean_key and provider != AI_CONFIG["provider"]:
        clean_key = _keychain_key(provider)
    if clean_key and provider == "openai" and parsed.hostname == "api.openai.com" and not clean_key.startswith("sk-"):
        raise ValueError("请输入真实的 OpenAI API Key（以 sk- 开头），不要粘贴 Netlify 隐藏变量提示文本")
    if clean_key:
        if provider == "gemini":
            _validate_gemini_key(clean_key)
        _save_keychain_key(provider, clean_key)
    AI_CONFIG.update({"enabled": enabled, "provider": provider, "base_url": base_url.rstrip("/"), "model": model.strip() or "gpt-image-2", "api_key": clean_key})
    _save_keychain_config({"provider": provider, "base_url": AI_CONFIG["base_url"], "model": AI_CONFIG["model"]})
    return public_ai_config()


def public_ai_config() -> dict:
    return {"enabled": AI_CONFIG["enabled"], "provider": AI_CONFIG["provider"], "base_url": AI_CONFIG["base_url"], "model": AI_CONFIG["model"], "configured": bool(AI_CONFIG["api_key"])}


def _validate_protected_pixels(
    generated: Image.Image,
    request_image: Image.Image,
    protected_mask: Image.Image | None,
) -> None:
    """Reject generations that recolor, redraw, move, or reframe supplied pixels."""
    if protected_mask is None:
        return
    candidate = generated.convert("RGB").resize(request_image.size, Image.Resampling.LANCZOS)
    original = np.asarray(request_image.convert("RGB"), dtype=np.int16)
    result = np.asarray(candidate, dtype=np.int16)
    protected = np.asarray(protected_mask.getchannel("A")) >= 250
    if not np.any(protected):
        return
    differences = np.abs(original - result).mean(axis=2)[protected]
    mean_difference = float(differences.mean())
    p95_difference = float(np.percentile(differences, 95))
    changed_ratio = float(np.mean(differences > 18))
    original_protected = original[protected]
    result_protected = result[protected]
    channel_shift = np.abs(result_protected.mean(axis=0) - original_protected.mean(axis=0))
    max_channel_shift = float(channel_shift.max())
    # Product colour is a hard constraint. Only tiny JPEG quantisation noise is
    # tolerated; colour grading, white-balance shifts and regenerated material
    # are rejected even when geometry appears unchanged.
    if mean_difference > 9 or p95_difference > 28 or changed_ratio > 0.08 or max_channel_shift > 6:
        raise OutpaintError(
            "主体颜色/细节保护校验未通过：AI 对鞋子或人物进行了调色、重绘或重新构图"
            f"（平均差异 {mean_difference:.1f}，最大通道色偏 {max_channel_shift:.1f}，变化像素 {changed_ratio:.1%}）"
        )


def _foreground_validation_mask(
    source_layer: Image.Image,
    paste_xy: tuple[int, int],
    target_size: tuple[int, int],
    protected_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Protect likely product/person pixels while leaving background editable."""
    source = np.asarray(source_layer.convert("RGB"), dtype=np.int16)
    height, width = source.shape[:2]
    band = max(2, round(min(width, height) * .025))
    border = np.concatenate((
        source[:band].reshape(-1, 3), source[-band:].reshape(-1, 3),
        source[:, :band].reshape(-1, 3), source[:, -band:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    border_noise = np.linalg.norm(border - background, axis=1)
    threshold = max(18.0, float(np.percentile(border_noise, 80)) * 1.8)
    foreground = np.linalg.norm(source - background, axis=2) > threshold
    if protected_box:
        bx, by, bw, bh = protected_box
        region = np.zeros((height, width), dtype=bool)
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(width, bx + bw), min(height, by + bh)
        region[y1:y2, x1:x2] = True
        foreground &= region
    # Remove isolated background noise and slightly dilate to protect fine shoe,
    # garment and skin edges without freezing the surrounding floor/background.
    subject = Image.fromarray((foreground * 255).astype(np.uint8))
    subject = subject.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.MaxFilter(7))
    if np.mean(np.asarray(subject) > 0) < .005:
        # Fail safely for unusually low-contrast material: protect the supplied
        # merchandise region instead of silently allowing product recolouring.
        subject = Image.new("L", (width, height), 0)
        if protected_box:
            bx, by, bw, bh = protected_box
            subject.paste(255, (max(0, bx), max(0, by), min(width, bx + bw), min(height, by + bh)))
        else:
            subject.paste(255, (0, 0, width, height))
    canvas = Image.new("RGBA", target_size, (255, 255, 255, 0))
    alpha = Image.new("RGBA", subject.size, (255, 255, 255, 0))
    alpha.putalpha(subject)
    canvas.paste(alpha, paste_xy, alpha)
    return canvas


def _request_work_size(provider: str, target_size: tuple[int, int]) -> tuple[int, int]:
    if target_size[1] <= target_size[0]:
        return (1024, 1024)
    # Gemini is explicitly asked for 3:4 output, so its input canvas must also
    # be 3:4. A 2:3 input made the model zoom/recenter or return the full source.
    return (1536, 2048) if provider == "gemini" else (1024, 1536)


def _gemini_outpaint(
    request_image: Image.Image,
    target_size: tuple[int, int],
    prompt: str,
    protected_mask: Image.Image | None = None,
) -> Image.Image:
    stream = io.BytesIO()
    request_image.save(stream, "PNG")
    aspect = "3:4" if target_size[1] > target_size[0] else "1:1"
    model_name = AI_CONFIG["model"].lower()
    image_size = "1K" if "lite" in model_name else "4K" if "pro-image" in model_name else "2K"
    payload = {
        "model": AI_CONFIG["model"],
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": "image/png", "data": base64.b64encode(stream.getvalue()).decode()},
        ],
        # Gemini Interactions currently accepts JPEG for generated image output.
        # The transparent PNG remains the input canvas used to mark missing areas.
        "response_format": {"type": "image", "mime_type": "image/jpeg", "aspect_ratio": aspect, "image_size": image_size},
    }
    endpoint = f'{AI_CONFIG["base_url"].rstrip("/")}/interactions'
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": AI_CONFIG["api_key"]}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            error = body.get("error", body)
            detail = error.get("message") or error.get("status") if isinstance(error, dict) else str(error)
        except Exception:
            detail = f"{exc.reason}"
        raise OutpaintError(f"Gemini 请求失败（HTTP {exc.code}）：{detail}") from exc
    encoded = _gemini_image_data(result)
    if not encoded:
        step_types = [step.get("type", "unknown") for step in result.get("steps", []) if isinstance(step, dict)]
        detail = f"；响应步骤：{', '.join(step_types)}" if step_types else ""
        raise OutpaintError(f"Gemini 响应中没有可用的内嵌图片{detail}")
    generated = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    _validate_protected_pixels(generated, request_image, protected_mask)
    return generated.resize(target_size, Image.Resampling.LANCZOS)


def _gemini_image_data(result: dict) -> str:
    """Read either SDK-style convenience output or raw REST model-output steps."""
    direct = result.get("output_image")
    if isinstance(direct, dict) and direct.get("data"):
        return direct["data"]
    for step in reversed(result.get("steps", [])):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for block in reversed(step.get("content", [])):
            if isinstance(block, dict) and block.get("type") == "image" and block.get("data"):
                return block["data"]
    return ""


def _named_png(image: Image.Image, name: str) -> io.BytesIO:
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)
    stream.name = name
    return stream


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    if "401" in lower or "authentication" in lower or "api key" in lower:
        return "API Key 无效或无权限，请在 AI 设置中重新填写"
    if "429" in lower or "quota" in lower or "rate limit" in lower:
        return "AI 额度不足或请求过快，请检查余额后重试"
    if "model" in lower and ("not found" in lower or "unsupported" in lower or "invalid" in lower):
        return "当前模型不支持图片编辑/蒙版扩图，请更换 Images Edit 模型"
    if "connect" in lower or "timeout" in lower:
        return "无法连接 AI 服务，请检查 API 地址、网络或本地模型网关"
    return message


def gpt_outpaint(
    source_layer: Image.Image,
    paste_xy: tuple[int, int],
    target_size: tuple[int, int],
    protected_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Outpaint one coherent, sharp full frame without tiled or blurred fills."""
    if not AI_CONFIG["enabled"] or not AI_CONFIG["api_key"]:
        raise OutpaintError("未配置可用的 AI 扩图服务")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OutpaintError("未安装 openai Python SDK") from exc

    work_size = _request_work_size(AI_CONFIG["provider"], target_size)
    composed = Image.new("RGBA", target_size, (0, 0, 0, 0))
    composed.paste(source_layer.convert("RGBA"), paste_xy)
    mask = Image.new("RGBA", target_size, (255, 255, 255, 0))
    x, y = paste_xy
    left, top = max(0, x), max(0, y)
    right, bottom = min(target_size[0], x + source_layer.width), min(target_size[1], y + source_layer.height)
    mask.paste((255, 255, 255, 255), (left, top, right, bottom))
    # Let the model repaint a narrow overlap inside the source boundary. Keeping
    # the entire source opaque freezes its rectangular edge into the result.
    overlap = max(12, round(min(target_size) * .035))
    if left > 0:
        mask.paste((255, 255, 255, 0), (left, top, min(right, left + overlap), bottom))
    if right < target_size[0]:
        mask.paste((255, 255, 255, 0), (max(left, right - overlap), top, right, bottom))
    if top > 0:
        mask.paste((255, 255, 255, 0), (left, top, right, min(bottom, top + overlap)))
    if bottom < target_size[1]:
        mask.paste((255, 255, 255, 0), (left, max(top, bottom - overlap), right, bottom))
    request_image = composed.resize(work_size, Image.Resampling.LANCZOS)
    request_mask = mask.resize(work_size, Image.Resampling.NEAREST)
    validation_mask = _foreground_validation_mask(source_layer, paste_xy, target_size, protected_box)
    validation_mask = validation_mask.resize(work_size, Image.Resampling.NEAREST)
    prompt = (
        "Outpaint this into one complete, coherent, sharp professional ecommerce photograph. "
        "The supplied canvas geometry and image placement are locked: do not zoom, crop, recenter, reframe, rotate, or return the original full photograph. "
        "Generate the transparent missing surroundings by extending or naturally rebuilding the background, floor, lighting, perspective and texture. "
        "The background outside the product/person may change freely when needed for a coherent outpaint. "
        "Do not blur, tile, mirror, collage, splice, duplicate, or create visible seams. "
        "Do not add a second product or person. Do not change any protected source pixel. "
        "Do not color grade, relight, change exposure, change white balance, alter saturation, or shift hue on the supplied image. "
        "Preserve every shoe, leg, person, garment, skin tone, logo, material, stitch, buckle, sole, color, proportion, pose and product detail exactly."
    )
    try:
        if AI_CONFIG["provider"] == "gemini":
            return _gemini_outpaint(request_image, target_size, prompt, validation_mask)
        client = OpenAI(api_key=AI_CONFIG["api_key"], base_url=AI_CONFIG["base_url"])
        result = client.images.edit(
            model=AI_CONFIG["model"], image=_named_png(request_image, "input.png"),
            mask=_named_png(request_mask, "mask.png"), prompt=prompt,
            size="1024x1536" if work_size[1] > work_size[0] else "1024x1024",
            quality="medium", output_format="png",
        )
        generated = Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json))).convert("RGB")
        _validate_protected_pixels(generated, request_image, validation_mask)
    except Exception as exc:
        raise OutpaintError(f"AI 扩图失败：{_friendly_error(exc)}") from exc
    return generated.resize(target_size, Image.Resampling.LANCZOS)
