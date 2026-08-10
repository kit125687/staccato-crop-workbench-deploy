from __future__ import annotations

import io
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .core import OUTPUT_SIZES, ImageItem, assign_barcodes, export_items, output_layout, render_output, scan_root

NO_AI_MODE = os.getenv("NO_AI_MODE", "false").lower() == "true"
PUBLIC_CLOUD = os.getenv("PUBLIC_CLOUD", "false").lower() == "true"
if not NO_AI_MODE:
    from .outpaint import configure_ai, public_ai_config

app = FastAPI(title="规范切图工作台 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://staccato-video-workbench.netlify.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
JOBS: dict[str, dict] = {}
IMAGE_INDEX: dict[str, ImageItem] = {}
PREVIEW_CACHE: dict[tuple, bytes] = {}
CLOUD_ROOT = Path(os.getenv("CLOUD_JOB_ROOT", "/tmp/staccato-crop-jobs"))
CLOUD_ROOT.mkdir(parents=True, exist_ok=True)


class ScanRequest(BaseModel):
    root: str


class OrderRequest(BaseModel):
    image_ids: list[str]


class ClassificationRequest(BaseModel):
    image_type: str


class CropRequest(BaseModel):
    offset_x: int = 0
    offset_y: int = 0
    zoom: int = 100


class OutputSizeRequest(BaseModel):
    barcode: str
    width: int
    height: int


class OutputNameRequest(BaseModel):
    barcode: str
    filename: str


class ExportRequest(BaseModel):
    completion_mode: str = "ai"


class AiConfigRequest(BaseModel):
    enabled: bool = True
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-image-2"
    api_key: str = ""


def public_job(job: dict) -> dict:
    return {"id": job["id"], "root": job["root"], "status": job["status"], "errors": job["errors"], "cloud": job.get("cloud", False), "items": [item.public() for item in job["items"]]}


def _clean_cloud_jobs(max_age: int = 7200) -> None:
    cutoff = time.time() - max_age
    for folder in CLOUD_ROOT.iterdir():
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder)
        except OSError:
            continue


def _safe_upload_path(relative: str) -> Path:
    parts = [part for part in relative.replace("\\", "/").split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise HTTPException(400, "上传文件路径无效")
    return Path(*parts)


@app.get("/api/health")
def health():
    if NO_AI_MODE:
        return {"ok": True, "no_ai_mode": True, "ai_configured": False, "ai": None}
    config = public_ai_config()
    return {"ok": True, "no_ai_mode": False, "ai_configured": config["configured"], "ai": config}


@app.get("/api/settings/ai")
def get_ai_settings():
    if NO_AI_MODE:
        raise HTTPException(404, "无AI留白版不提供AI设置")
    return public_ai_config()


@app.put("/api/settings/ai")
def update_ai_settings(request: AiConfigRequest):
    if PUBLIC_CLOUD:
        raise HTTPException(403, "公开版AI配置由服务端管理")
    if NO_AI_MODE:
        raise HTTPException(404, "无AI留白版不提供AI设置")
    try:
        return configure_ai(request.enabled, request.base_url, request.model, request.api_key, request.provider)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/jobs/scan")
def scan(request: ScanRequest):
    try:
        items, errors = scan_root(request.root)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "root": str(Path(request.root).expanduser()), "status": "review", "errors": errors, "items": items}
    JOBS[job_id] = job
    for item in items:
        IMAGE_INDEX[item.id] = item
    return public_job(job)


@app.post("/api/cloud/jobs/scan")
async def cloud_scan(files: list[UploadFile] = File(...), paths: list[str] = Form(...)):
    if len(files) != len(paths):
        raise HTTPException(400, "文件与相对路径数量不一致")
    if not files or len(files) > 200:
        raise HTTPException(400, "每批需上传 1–200 张图片")
    _clean_cloud_jobs()
    job_id = uuid.uuid4().hex[:12]
    upload_root = CLOUD_ROOT / job_id / "input"
    upload_root.mkdir(parents=True)
    total = 0
    try:
        for upload, relative in zip(files, paths):
            suffix = Path(upload.filename or relative).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                continue
            payload = await upload.read()
            total += len(payload)
            if len(payload) > 25_000_000 or total > 350_000_000:
                raise HTTPException(413, "单图不能超过25MB，整批不能超过350MB")
            target = upload_root / _safe_upload_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        items, errors = scan_root(str(upload_root))
    except Exception:
        shutil.rmtree(CLOUD_ROOT / job_id, ignore_errors=True)
        raise
    job = {"id": job_id, "root": str(upload_root), "status": "review", "errors": errors, "items": items, "cloud": True}
    JOBS[job_id] = job
    for item in items:
        IMAGE_INDEX[item.id] = item
    return public_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "批次不存在")
    return public_job(JOBS[job_id])


@app.put("/api/jobs/{job_id}/order")
def update_order(job_id: str, request: OrderRequest):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "批次不存在")
    current = {item.id: item for item in job["items"]}
    if set(request.image_ids) != set(current):
        raise HTTPException(400, "排序列表与批次图片不一致")
    job["items"] = [current[item_id] for item_id in request.image_ids]
    assign_barcodes(job["items"])
    return public_job(job)


@app.put("/api/images/{image_id}/classification")
def update_classification(image_id: str, request: ClassificationRequest):
    if request.image_type not in {"静物", "腿模", "全身"}:
        raise HTTPException(400, "无效图片类型")
    item = IMAGE_INDEX.get(image_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    item.image_type = request.image_type
    item.confidence = 1
    item.reason = "人工确认"
    item.needs_review = False
    item.review_reasons = []
    for job in JOBS.values():
        if item in job["items"]:
            PREVIEW_CACHE.clear()
            assign_barcodes(job["items"])
            return public_job(job)
    raise HTTPException(404, "所属批次不存在")


@app.put("/api/images/{image_id}/crop")
def update_crop(image_id: str, request: CropRequest):
    item = IMAGE_INDEX.get(image_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    item.crop = {"offset_x": max(-1400, min(1400, request.offset_x)), "offset_y": max(-1600, min(1600, request.offset_y)), "zoom": max(35, min(240, request.zoom))}
    return item.public()


@app.put("/api/images/{image_id}/output-size")
def update_output_size(image_id: str, request: OutputSizeRequest):
    item = IMAGE_INDEX.get(image_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    if request.barcode not in item.barcodes:
        raise HTTPException(400, "该图片没有此输出编号")
    if not 320 <= request.width <= 8000 or not 320 <= request.height <= 8000:
        raise HTTPException(400, "宽高需在 320–8000 像素之间")
    if request.width * request.height > 36_000_000:
        raise HTTPException(400, "自定义画布不能超过 3600 万像素")
    item.output_sizes[request.barcode] = [request.width, request.height]
    PREVIEW_CACHE.clear()
    return item.public()


@app.put("/api/images/{image_id}/output-name")
def update_output_name(image_id: str, request: OutputNameRequest):
    item = IMAGE_INDEX.get(image_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    if request.barcode not in item.barcodes:
        raise HTTPException(400, "该图片没有此输出编号")
    filename = request.filename.strip()
    if not filename.lower().endswith((".jpg", ".jpeg")):
        filename += ".jpg"
    if not filename or filename in {".jpg", ".jpeg"} or any(char in filename for char in "/\\:\0"):
        raise HTTPException(400, "文件名不能为空，且不能包含 / \\ : 等字符")
    if len(filename) > 180:
        raise HTTPException(400, "文件名不能超过 180 个字符")
    parent = Path(item.path).parent
    for other in IMAGE_INDEX.values():
        if other.id == item.id or Path(other.path).parent != parent:
            continue
        for code in other.barcodes:
            other_name = other.output_names.get(code) or f"{other.folder}_{code}.jpg"
            if other_name.casefold() == filename.casefold():
                raise HTTPException(400, f"文件名已被同商品的 _{code} 输出使用")
    item.output_names[request.barcode] = filename
    return item.public()


@app.get("/api/images/{image_id}/preview")
def preview(image_id: str, barcode: Optional[str] = None, size: int = 720, completion_mode: str = "ai"):
    item = IMAGE_INDEX.get(image_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    selected = barcode or (item.barcodes[0] if item.barcodes else "43")
    size = max(120, min(1000, size))
    completion_mode = "blank" if completion_mode == "blank" else "ai"
    key = (image_id, selected, tuple(item.output_sizes.get(selected, [])), item.crop["offset_x"], item.crop["offset_y"], item.crop["zoom"], size, completion_mode)
    if key in PREVIEW_CACHE:
        return Response(PREVIEW_CACHE[key], media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
    try:
        image, _ = render_output(item, selected, allow_complex_fallback=True, preview_max=size, completion_mode=completion_mode)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    stream = io.BytesIO()
    image.save(stream, "JPEG", quality=86, optimize=True)
    payload = stream.getvalue()
    if len(PREVIEW_CACHE) > 300:
        PREVIEW_CACHE.clear()
    PREVIEW_CACHE[key] = payload
    return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/images/{image_id}/layout")
def preview_layout(image_id: str, barcode: Optional[str] = None):
    item = IMAGE_INDEX.get(image_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    selected = barcode or (item.barcodes[0] if item.barcodes else "43")
    if selected not in OUTPUT_SIZES:
        raise HTTPException(400, "无效输出编号")
    return output_layout(item, selected)


@app.post("/api/jobs/{job_id}/process")
def process(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "批次不存在")
    # Background completion is automatic during export and is not a manual
    # review condition. Only genuine classification ambiguity stays reviewable.
    job["status"] = "ready"
    return public_job(job)


@app.post("/api/jobs/{job_id}/export")
def export(job_id: str, request: ExportRequest):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "批次不存在")
    completion_mode = "blank" if request.completion_mode == "blank" else "ai"
    exported, failed = export_items(job["items"], completion_mode=completion_mode)
    job["status"] = "completed" if not failed else "partial"
    response = {"job": public_job(job), "exported": exported, "failed": failed}
    if job.get("cloud") and exported:
        archive = CLOUD_ROOT / job_id / "staccato-crops.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for result in exported:
                output = Path(result["path"])
                try:
                    arcname = output.relative_to(Path(job["root"]))
                except ValueError:
                    arcname = Path(output.parent.name) / output.name
                bundle.write(output, arcname.as_posix())
        response["download_url"] = f"/api/cloud/jobs/{job_id}/download"
    return response


@app.get("/api/cloud/jobs/{job_id}/download")
def download_cloud_export(job_id: str):
    job = JOBS.get(job_id)
    archive = CLOUD_ROOT / job_id / "staccato-crops.zip"
    if not job or not job.get("cloud") or not archive.exists():
        raise HTTPException(404, "导出压缩包不存在或已过期")
    return FileResponse(archive, media_type="application/zip", filename=f"staccato-crops-{job_id}.zip")
