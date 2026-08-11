from __future__ import annotations

import base64
import io
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageChops, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OUTPUT_CODES = {str(n) for n in list(range(42, 57)) + list(range(60, 71))}
OUTPUT_SIZES = {code: (3000, 4000) for code in OUTPUT_CODES}
OUTPUT_SIZES.update({"45": (3000, 3000), "50": (3000, 3000)})
STATIC_CODES = [["43"], ["44"], ["42"]]
LEG_CODES = [["45", "46"], ["48"], ["49"], ["47"]]
FULL_CODES = [["50", "51"], ["52"], ["56"], ["53"], ["54"], ["55"]]
SHARED_CODES = [str(n) for n in range(60, 71)]


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class ImageItem:
    id: str
    folder: str
    path: str
    filename: str
    width: int
    height: int
    image_type: str
    confidence: float
    reason: str
    subject_box: Box
    background: str
    needs_review: bool
    review_reasons: list[str] = field(default_factory=list)
    keypoints: list[dict] = field(default_factory=list)
    barcodes: list[str] = field(default_factory=list)
    crop: dict = field(default_factory=lambda: {"offset_x": 0, "offset_y": 0, "zoom": 100})
    output_sizes: dict[str, list[int]] = field(default_factory=dict)
    output_names: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict:
        data = asdict(self)
        data["preview_url"] = f"/api/images/{self.id}/preview"
        return data


def _downsample(image: Image.Image, maximum: int = 480) -> np.ndarray:
    sample = image.copy().convert("RGB")
    sample.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    return np.asarray(sample).astype(np.int16)


def _foreground_box(image: Image.Image) -> tuple[Box, float]:
    arr = _downsample(image)
    h, w = arr.shape[:2]
    border = np.concatenate((arr[: max(2, h // 30)].reshape(-1, 3), arr[-max(2, h // 30):].reshape(-1, 3), arr[:, : max(2, w // 30)].reshape(-1, 3), arr[:, -max(2, w // 30):].reshape(-1, 3)))
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(arr - bg, axis=2)
    threshold = max(24.0, float(np.percentile(distance, 65)))
    mask = distance > threshold
    ys, xs = np.where(mask)
    if len(xs) < w * h * 0.015:
        return Box(int(image.width * .1), int(image.height * .1), int(image.width * .8), int(image.height * .8)), .28
    pad = max(2, min(w, h) // 45)
    x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
    y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
    sx, sy = image.width / w, image.height / h
    return Box(round(x1 * sx), round(y1 * sy), max(1, round((x2 - x1) * sx)), max(1, round((y2 - y1) * sy))), min(.92, .48 + len(xs) / (w * h))


def _skin_structure(image: Image.Image) -> tuple[float, float]:
    """Return skin-like coverage for the whole image and its upper 30%."""
    arr = _downsample(image).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maximum, minimum = arr.max(axis=2), arr.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nz = delta > 1
    red = nz & (maximum == r)
    green = nz & (maximum == g)
    blue = nz & (maximum == b)
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6
    hue[green] = (b[green] - r[green]) / delta[green] + 2
    hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4
    hue *= 30
    saturation = np.zeros_like(maximum)
    np.divide(delta * 255, maximum, out=saturation, where=maximum > 0)
    skin = (hue < 25) & (saturation > 35) & (saturation < 190) & (maximum > 70)
    top = skin[: max(1, round(skin.shape[0] * .30))]
    return float(skin.mean()), float(top.mean())


def _vision_body_pose(path: Path) -> list[dict]:
    """Use macOS Vision locally; return normalized top-left-origin points."""
    try:
        import Vision
        from Foundation import NSURL

        request = Vision.VNDetectHumanBodyPoseRequest.new()
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), {})
        ok, _ = handler.performRequests_error_([request], None)
        results = request.results() or []
        if not ok or not results:
            return []
        points, _ = results[0].recognizedPointsForJointsGroupName_error_(
            Vision.VNHumanBodyPoseObservationJointsGroupNameAll, None
        )
        names = {
            "nose": Vision.VNHumanBodyPoseObservationJointNameNose,
            "neck": Vision.VNHumanBodyPoseObservationJointNameNeck,
            "left_shoulder": Vision.VNHumanBodyPoseObservationJointNameLeftShoulder,
            "right_shoulder": Vision.VNHumanBodyPoseObservationJointNameRightShoulder,
            "left_hip": Vision.VNHumanBodyPoseObservationJointNameLeftHip,
            "right_hip": Vision.VNHumanBodyPoseObservationJointNameRightHip,
            "left_knee": Vision.VNHumanBodyPoseObservationJointNameLeftKnee,
            "right_knee": Vision.VNHumanBodyPoseObservationJointNameRightKnee,
            "left_ankle": Vision.VNHumanBodyPoseObservationJointNameLeftAnkle,
            "right_ankle": Vision.VNHumanBodyPoseObservationJointNameRightAnkle,
        }
        detected = []
        for label, key in names.items():
            point = points.get(key)
            if point and point.confidence() >= .25:
                location = point.location()
                detected.append({"name": label, "x": round(float(location.x), 4), "y": round(1 - float(location.y), 4), "confidence": round(float(point.confidence()), 3)})
        return detected
    except Exception:
        return []


def _mediapipe_body_pose(image: Image.Image) -> list[dict]:
    """Detect real human pose landmarks on Linux without an external API."""
    model_path = Path(os.getenv("POSE_MODEL_PATH", "/app/backend/models/pose_landmarker_lite.task"))
    if not model_path.is_file():
        return []
    try:
        import mediapipe as mp

        sample = image.copy().convert("RGB")
        sample.thumbnail((960, 960), Image.Resampling.LANCZOS)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=.45,
            min_pose_presence_confidence=.45,
        )
        names = {
            "nose": 0,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
            "left_heel": 29, "right_heel": 30,
            "left_foot": 31, "right_foot": 32,
        }
        frame = np.ascontiguousarray(np.asarray(sample, dtype=np.uint8))
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
        if not result.pose_landmarks:
            return []
        detected = []
        for label, index in names.items():
            best = max(
                (pose[index] for pose in result.pose_landmarks),
                key=lambda point: float(point.visibility or 0) * float(point.presence or 0),
            )
            confidence = min(float(best.visibility or 0), float(best.presence or 0))
            if confidence >= .30 and -.08 <= best.x <= 1.08 and -.08 <= best.y <= 1.08:
                detected.append({
                    "name": label,
                    "x": round(max(0., min(1., float(best.x))), 4),
                    "y": round(max(0., min(1., float(best.y))), 4),
                    "confidence": round(confidence, 3),
                })
        return detected
    except Exception:
        return []


def _background_type(image: Image.Image, box: Box) -> str:
    sample = _downsample(image)
    h, w = sample.shape[:2]
    edge = np.concatenate((sample[: h // 12].reshape(-1, 3), sample[-h // 12:].reshape(-1, 3), sample[:, : w // 12].reshape(-1, 3), sample[:, -w // 12:].reshape(-1, 3)))
    std = float(edge.std(axis=0).max())
    top, bottom = sample[: h // 10].mean(axis=(0, 1)), sample[-h // 10:].mean(axis=(0, 1))
    if std < 18:
        return "solid"
    if std < 42 and 10 < float(np.abs(top - bottom).mean()) < 90:
        return "gradient"
    return "complex"


def _shoe_subject_box(frame: Box, image: Image.Image, image_type: str) -> Box:
    """Approximate the merchandise area; for leg-model images ignore upper anatomy."""
    if image_type != "腿模":
        return frame
    y1 = max(frame.y, round(image.height * .43))
    y2 = min(image.height, frame.y2)
    if y2 - y1 < image.height * .15:
        y1 = max(frame.y, round(frame.y + frame.h * .48))
    return Box(frame.x, y1, frame.w, max(1, y2 - y1))


def _normalized_box(value: object, image: Image.Image) -> Box | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x, y, w, h = [max(0, min(1000, int(number))) for number in value]
    except (TypeError, ValueError):
        return None
    if w < 15 or h < 15:
        return None
    x2, y2 = min(1000, x + w), min(1000, y + h)
    return Box(
        round(x * image.width / 1000), round(y * image.height / 1000),
        max(1, round((x2 - x) * image.width / 1000)),
        max(1, round((y2 - y) * image.height / 1000)),
    )


def analyze_image(path: Path, folder: str) -> ImageItem:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        frame_box, subject_score = _foreground_box(image)
        skin_ratio, top_skin_ratio = _skin_structure(image)
        pose_candidate = skin_ratio < .60 and (top_skin_ratio >= .08 or frame_box.y <= image.height * .14)
        keypoints = _vision_body_pose(path) if pose_candidate else []
        if not keypoints and os.getenv("ENABLE_LOCAL_POSE", "true").lower() == "true":
            keypoints = _mediapipe_body_pose(image)
        names = {point["name"] for point in keypoints}
        shoulders = names & {"left_shoulder", "right_shoulder"}
        lower_body = names & {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
        pose_span = max((point["y"] for point in keypoints), default=0) - min((point["y"] for point in keypoints), default=0)
        upper_body = (
            len(keypoints) >= 4
            and ("neck" in names or len(shoulders) == 2)
            and bool(lower_body)
            and pose_span >= .25
            and skin_ratio < .60
        )
        if upper_body:
            image_type = "全身"
            confidence = min(.98, .84 + len(keypoints) * .012)
            reason = "人体关键点已达到肩颈或胸部以上，按全身图处理"
        elif len(lower_body) >= 2:
            image_type = "腿模"
            confidence = min(.97, .82 + len(lower_body) * .018)
            reason = "检测到真人髋、膝、踝或足部连续姿态，且躯体未达到胸部"
        elif top_skin_ratio >= .20 and skin_ratio < .60:
            image_type = "腿模"
            confidence = min(.94, .76 + top_skin_ratio * .35)
            reason = "检测到从画面上方延伸的腿部肤色结构，且未达到胸部"
        else:
            image_type = "静物"
            confidence = min(.97, .80 + subject_score * .16)
            reason = "未检测到人体关键点或连续腿部结构，按纯鞋静物处理"
        ai_result = None
        try:
            from .outpaint import analyze_product_image
            ai_result = analyze_product_image(image)
        except Exception:
            ai_result = None
        if ai_result:
            image_type = ai_result["image_type"]
            confidence = max(.0, min(.99, float(ai_result.get("confidence", .86))))
            reason = f'AI视觉识别：{ai_result.get("reason", "已按可见人体结构判断")}'
        review_reasons = []
        if len(keypoints) >= 3 and not upper_body and image_type == "静物":
            review_reasons.append("检测到零散人体关键点，需确认是否为产品细节误识别")
        if confidence < .8:
            review_reasons.append("分类置信度低")
        background = _background_type(image, frame_box)
        touches_edge = (
            frame_box.x < image.width * .015 or frame_box.x2 > image.width * .985
            or frame_box.y < image.height * .015 or frame_box.y2 > image.height * .985
        )
        # Edge contact is a completion trigger, not a review error. Non-uniform
        # backgrounds must use coherent AI outpainting instead of a visible fill.
        if touches_edge and background != "solid":
            background = "complex"
        ai_shoe_box = _normalized_box(ai_result.get("shoe_box"), image) if ai_result else None
        ai_person_box = _normalized_box(ai_result.get("person_box"), image) if ai_result else None
        if image_type == "全身":
            box = ai_person_box or frame_box
        else:
            box = ai_shoe_box or _shoe_subject_box(frame_box, image, image_type)
        return ImageItem(
            id=uuid.uuid4().hex[:12], folder=folder, path=str(path), filename=path.name,
            width=image.width, height=image.height, image_type=image_type,
            confidence=round(confidence, 3), reason=reason, subject_box=box,
            background=background, needs_review=bool(review_reasons), review_reasons=review_reasons,
            keypoints=keypoints,
        )


def is_generated_output(path: Path, folder_name: str) -> bool:
    match = re.fullmatch(r"(?:" + re.escape(folder_name) + r"|条码)_(\d+)\.jpe?g", path.name, flags=re.IGNORECASE)
    return bool(match and match.group(1) in OUTPUT_CODES)


def scan_root(root: str) -> tuple[list[ImageItem], list[str]]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("根目录不存在或不可读取")
    folders = sorted((p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name.lower())
    if not folders:
        folders = [base]
    items, errors, work = [], [], []
    for folder in folders:
        files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_generated_output(p, folder.name)), key=lambda p: p.name.lower())
        for path in files:
            work.append((path, folder.name))
    configured_workers = max(1, int(os.getenv("SCAN_WORKERS", "3")))
    with ThreadPoolExecutor(max_workers=min(configured_workers, max(1, len(work)))) as pool:
        futures = [pool.submit(analyze_image, path, folder) for path, folder in work]
        for (path, folder), future in zip(work, futures):
            try:
                items.append(future.result())
            except Exception as exc:
                errors.append(f"{folder}/{path.name}: {exc}")
    assign_barcodes(items)
    return items, errors


def assign_barcodes(items: Iterable[ImageItem]) -> None:
    by_folder: dict[str, list[ImageItem]] = {}
    for item in items:
        by_folder.setdefault(item.folder, []).append(item)
    for folder_items in by_folder.values():
        counters = {"静物": 0, "腿模": 0, "全身": 0}
        shared_index = 0
        for item in folder_items:
            table = STATIC_CODES if item.image_type == "静物" else LEG_CODES if item.image_type == "腿模" else FULL_CODES
            index = counters[item.image_type]
            counters[item.image_type] += 1
            if index < len(table):
                item.barcodes = list(table[index])
            elif item.image_type in ("静物", "腿模") and shared_index < len(SHARED_CODES):
                item.barcodes = [SHARED_CODES[shared_index]]
                shared_index += 1
            else:
                item.barcodes = []
                item.needs_review = True
                if "可用输出编号已耗尽" not in item.review_reasons:
                    item.review_reasons.append("可用输出编号已耗尽")


def safe_zone(image_type: str, target: tuple[int, int]) -> Box:
    w, h = target
    if image_type == "全身":
        return Box(round(w * .08), round(h * .05), round(w * .84), round(h * .90))
    if abs(w / h - 1) < .02:
        return Box(round(w * 668 / 3000), round(h * 640 / 3000), round(w * 1664 / 3000), round(h * 1665 / 3000))
    return Box(round(w * 160 / 3000), round(h * 768 / 4000), round(w * 2680 / 3000), round(h * 2400 / 4000))


def _clean_canvas(image: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    """Create a crisp synthetic background; never stretch, tile, or blur the photo."""
    sample = _downsample(image, 720).astype(np.float32)
    band_y = max(2, sample.shape[0] // 24)
    band_x = max(2, sample.shape[1] // 24)
    edge = np.concatenate((
        sample[:band_y].reshape(-1, 3), sample[-band_y:].reshape(-1, 3),
        sample[:, :band_x].reshape(-1, 3), sample[:, -band_x:].reshape(-1, 3),
    ))
    if mode == "solid":
        return Image.new("RGB", size, tuple(np.median(edge, axis=0).round().astype(int)))
    top = sample[: max(2, sample.shape[0] // 18)].mean(axis=(0, 1))
    bottom = sample[-max(2, sample.shape[0] // 18):].mean(axis=(0, 1))
    y = np.linspace(0, 1, size[1], dtype=np.float32)[:, None, None]
    gradient = top[None, None, :] * (1 - y) + bottom[None, None, :] * y
    gradient = np.broadcast_to(gradient, (size[1], size[0], 3)).astype(np.uint8)
    return Image.fromarray(gradient, "RGB")


def _paste_seamless(canvas: Image.Image, image: Image.Image, xy: tuple[int, int]) -> None:
    """Feather only the outer background band so a scaled source has no rectangular seam."""
    width, height = image.size
    feather = max(8, round(min(width, height) * .045))
    x = np.arange(width)
    y = np.arange(height)
    distance = np.minimum.reduce((x, width - 1 - x))[None, :]
    distance = np.minimum(distance, np.minimum(y, height - 1 - y)[:, None])
    alpha = np.clip(distance / feather, 0, 1)
    # Smoothstep produces a gradual tonal transition without blurring the photo.
    alpha = alpha * alpha * (3 - 2 * alpha)
    mask = Image.fromarray(np.round(alpha * 255).astype(np.uint8))
    canvas.paste(image, xy, mask)


def _scaled_zone(image_type: str, original_target: tuple[int, int], target: tuple[int, int]) -> Box:
    zone = safe_zone(image_type, original_target)
    sx, sy = target[0] / original_target[0], target[1] / original_target[1]
    return Box(round(zone.x * sx), round(zone.y * sy), round(zone.w * sx), round(zone.h * sy))


def output_layout(item: ImageItem, barcode: str) -> dict:
    """Return the detected subject position in the rendered output canvas."""
    target = tuple(item.output_sizes.get(barcode, OUTPUT_SIZES[barcode]))
    zone = safe_zone(item.image_type, target)
    box = item.subject_box
    zoom = item.crop.get("zoom", 100) / 100
    scale = min(zone.w / max(1, box.w), zone.h / max(1, box.h)) * zoom
    subject_cx = (box.x + box.w / 2) * scale
    subject_cy = (box.y + box.h / 2) * scale
    paste_x = round(zone.x + zone.w / 2 + item.crop.get("offset_x", 0) - subject_cx)
    paste_y = round(zone.y + zone.h / 2 + item.crop.get("offset_y", 0) - subject_cy)
    detail_like = item.image_type == "静物" and (box.w * box.h) / max(1, item.width * item.height) >= .88
    if detail_like:
        scale = max(target[0] / item.width, target[1] / item.height) * zoom
        paste_x = round((target[0] - item.width * scale) / 2 + item.crop.get("offset_x", 0))
        paste_y = round((target[1] - item.height * scale) / 2 + item.crop.get("offset_y", 0))
    elif item.image_type == "腿模":
        preferred = round(target[1] * .025)
        min_y = round(zone.y - box.y * scale)
        max_y = round(zone.y2 - (box.y + box.h) * scale)
        paste_y = round((min_y + max_y) / 2) if max_y < min_y else max(min_y, min(preferred, max_y))
        paste_y += item.crop.get("offset_y", 0)
    elif item.image_type == "全身":
        scale = min(target[0] * .84 / item.width, target[1] * .90 / item.height) * zoom
        paste_x = round((target[0] - item.width * scale) / 2 + item.crop.get("offset_x", 0))
        paste_y = round(target[1] * .05 + item.crop.get("offset_y", 0))
    subject = Box(
        round(paste_x + box.x * scale), round(paste_y + box.y * scale),
        max(1, round(box.w * scale)), max(1, round(box.h * scale)),
    )
    return {"target": list(target), "subject_box": asdict(subject), "safe_zone": asdict(zone)}


def render_output(item: ImageItem, barcode: str, allow_complex_fallback: bool = False, preview_max: Optional[int] = None, force_unextended: bool = False, completion_mode: str = "ai") -> tuple[Image.Image, str]:
    configured = item.output_sizes.get(barcode)
    original_target = tuple(configured) if configured else OUTPUT_SIZES[barcode]
    factor = min(1.0, preview_max / max(original_target)) if preview_max else 1.0
    target = (round(original_target[0] * factor), round(original_target[1] * factor))
    with Image.open(item.path) as raw:
        source = ImageOps.exif_transpose(raw).convert("RGB")
    zone = _scaled_zone(item.image_type, original_target, target)
    box = item.subject_box
    scale = min(zone.w / max(1, box.w), zone.h / max(1, box.h)) * item.crop.get("zoom", 100) / 100
    scaled_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    resized = source.resize(scaled_size, Image.Resampling.LANCZOS)
    subject_cx = (box.x + box.w / 2) * scale
    subject_cy = (box.y + box.h / 2) * scale
    target_cx = zone.x + zone.w / 2 + item.crop.get("offset_x", 0) * factor
    target_cy = zone.y + zone.h / 2 + item.crop.get("offset_y", 0) * factor
    paste_x, paste_y = round(target_cx - subject_cx), round(target_cy - subject_cy)
    detail_like = item.image_type == "静物" and (box.w * box.h) / max(1, item.width * item.height) >= .88
    if detail_like:
        # Product close-ups should fill the canvas instead of asking a model to
        # invent pixels that could change product materials or details.
        cover_scale = max(target[0] / source.width, target[1] / source.height) * item.crop.get("zoom", 100) / 100
        scale = cover_scale
        scaled_size = (max(1, round(source.width * cover_scale)), max(1, round(source.height * cover_scale)))
        resized = source.resize(scaled_size, Image.Resampling.LANCZOS)
        paste_x = round((target[0] - resized.width) / 2 + item.crop.get("offset_x", 0) * factor)
        paste_y = round((target[1] - resized.height) / 2 + item.crop.get("offset_y", 0) * factor)
    elif item.image_type == "腿模":
        preferred = round(target[1] * .025)
        min_y = round(zone.y - box.y * scale)
        max_y = round(zone.y2 - (box.y + box.h) * scale)
        paste_y = round((min_y + max_y) / 2) if max_y < min_y else max(min_y, min(preferred, max_y))
        paste_y += round(item.crop.get("offset_y", 0) * factor)
    elif item.image_type == "全身":
        fit_scale = min(target[0] * .84 / source.width, target[1] * .90 / source.height) * item.crop.get("zoom", 100) / 100
        scale = fit_scale
        scaled_size = (max(1, round(source.width * fit_scale)), max(1, round(source.height * fit_scale)))
        resized = source.resize(scaled_size, Image.Resampling.LANCZOS)
        paste_x = round((target[0] - resized.width) / 2 + item.crop.get("offset_x", 0) * factor)
        paste_y = round(target[1] * .05 + item.crop.get("offset_y", 0) * factor)
    covers = paste_x <= 0 and paste_y <= 0 and paste_x + resized.width >= target[0] and paste_y + resized.height >= target[1]
    if covers:
        canvas = Image.new("RGB", target)
        canvas.paste(resized, (paste_x, paste_y))
        return canvas, "direct"
    if completion_mode == "blank" or os.getenv("NO_AI_MODE", "false").lower() == "true":
        canvas = Image.new("RGB", target, (255, 255, 255))
        canvas.paste(resized, (paste_x, paste_y))
        return canvas, "no_ai_blank"
    if force_unextended:
        canvas = Image.new("RGB", target, (245, 245, 245))
        canvas.paste(resized, (paste_x, paste_y))
        return canvas, "fallback_unextended"
    if not allow_complex_fallback:
        from .outpaint import gpt_outpaint
        protected = None if item.image_type in {"腿模", "全身"} else (
            round(box.x * scale), round(box.y * scale),
            max(1, round(box.w * scale)), max(1, round(box.h * scale)),
        )
        return gpt_outpaint(resized, (paste_x, paste_y), target, protected), "ai_outpaint"
    canvas = _clean_canvas(source, target, item.background if item.background != "complex" else "gradient")
    _paste_seamless(canvas, resized, (paste_x, paste_y))
    method = "solid_extend" if item.background == "solid" else "gradient_extend"
    if item.background == "complex":
        method = "preview_only"
    return canvas, method


def image_to_data_url(image: Image.Image, maximum: int = 1000) -> str:
    preview = image.copy()
    preview.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    preview.save(stream, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def _jpeg_under_limit(image: Image.Image, maximum_bytes: int = 2_000_000) -> tuple[bytes, int]:
    """Encode at the highest practical JPEG quality while enforcing 2 MB."""
    best = b""
    best_quality = 45
    low, high = 45, 95
    while low <= high:
        quality = (low + high) // 2
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=quality, subsampling=0)
        payload = stream.getvalue()
        if len(payload) <= maximum_bytes:
            best, best_quality = payload, quality
            low = quality + 1
        else:
            high = quality - 1
    if not best:
        low, high = 1, 44
        while low <= high:
            quality = (low + high) // 2
            stream = io.BytesIO()
            image.save(stream, "JPEG", quality=quality, subsampling=2)
            payload = stream.getvalue()
            if len(payload) <= maximum_bytes:
                best, best_quality = payload, quality
                low = quality + 1
            else:
                high = quality - 1
    if not best:
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=1, subsampling=2)
        best, best_quality = stream.getvalue(), 1
    if len(best) > maximum_bytes:
        raise ValueError("当前像素尺寸无法在不缩小画布的情况下压到 2MB 内")
    return best, best_quality


def _output_path(item: ImageItem, barcode: str) -> Path:
    name = item.output_names.get(barcode) or f"{item.folder}_{barcode}.jpg"
    return Path(item.path).parent / name


def _export_one(item: ImageItem, barcode: str, completion_mode: str = "ai") -> tuple[dict | None, dict | None]:
    try:
        image, method = render_output(item, barcode, completion_mode=completion_mode)
        output = _output_path(item, barcode)
        payload, quality = _jpeg_under_limit(image)
        output.write_bytes(payload)
        return {"path": str(output), "barcode": barcode, "method": method, "size": list(image.size), "bytes": len(payload), "quality": quality}, None
    except Exception as exc:
        try:
            image, _ = render_output(item, barcode, force_unextended=True, completion_mode=completion_mode)
            output = _output_path(item, barcode)
            payload, quality = _jpeg_under_limit(image)
            output.write_bytes(payload)
            return {"path": str(output), "barcode": barcode, "method": "fallback_unextended", "size": list(image.size), "bytes": len(payload), "quality": quality, "warning": str(exc), "filename": item.filename}, None
        except Exception as fallback_exc:
            return None, {"image_id": item.id, "filename": item.filename, "barcode": barcode, "error": f"AI：{exc}；本地兜底：{fallback_exc}"}


def export_items(items: Iterable[ImageItem], quality: int = 95, completion_mode: str = "ai") -> tuple[list[dict], list[dict]]:
    tasks = [(item, barcode) for item in items for barcode in item.barcodes]
    exported, failed = [], []
    # Image generation and JPEG encoding are independent per output. A small
    # worker pool improves batch speed without flooding paid model endpoints.
    configured_workers = max(1, int(os.getenv("EXPORT_WORKERS", "3")))
    with ThreadPoolExecutor(max_workers=min(configured_workers, max(1, len(tasks)))) as pool:
        futures = [pool.submit(_export_one, item, barcode, completion_mode) for item, barcode in tasks]
        for future in futures:
            success, error = future.result()
            if success:
                exported.append(success)
            if error:
                failed.append(error)
    return exported, failed
