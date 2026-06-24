"""
TraceMark - AI 内容水印系统 v2.4 (AI防伪检测版)
"""
import os
import uuid
import urllib.request
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ✅ 自动下载 U²-Net 模型（如果不存在）
MODEL_PATH = Path(__file__).parent / "watermarking" / "u2net.pth"
MODEL_URL = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"

if not MODEL_PATH.exists():
    print(f"[📥] 正在下载 U²-Net 模型...")
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print(f"[✅] 模型下载完成: {MODEL_PATH}")
    except Exception as e:
        print(f"[⚠️] 模型下载失败: {e}")
        print(f"[ℹ️] 将使用默认策略，某些功能可能受限")

from watermarking.lsb import embed_uid_in_image as lsb_embed, extract_uid_from_image as lsb_extract
from watermarking.dct import embed_uid_in_image as dct_embed, extract_uid_from_image as dct_extract
from watermarking.llm_strategist import WatermarkStrategist  # ✅ 智能策略
from watermarking.tamper_detector import detect_tampering  # ✅ 新增：AI防伪检测
from watermarking.adaptive_watermark import calculate_adaptive_strength, embed_adaptive_watermark, STRENGTH_LEVELS  # ✅ 新增：自适应水印
from database import log_watermark_operation, get_watermark_history, get_watermark_stats, clear_watermark_history  # ✅ 新增：数据库操作

app = FastAPI(
    title="TraceMark - AI内容水印系统",
    version="2.4",
    description="LLM智能决策 + AI防伪检测 + U²-Net/LSB/DCT混合架构水印系统"
)

BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp_files"
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ✅ 全局单例，避免每次请求重建客户端
strategist = WatermarkStrategist()


def safe_unlink(path: Path):
    """安全删除临时文件"""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/watermark", summary="智能嵌入水印")
async def add_watermark(
        file: UploadFile = File(...),
        uid: str = Form(..., description="唯一用户ID")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式")

    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"input_{unique_id}{ext}"
    output_path = TEMP_DIR / f"watermarked_{unique_id}.png"

    try:
        # 1. 保存上传文件
        content = await file.read()
        input_path.write_bytes(content)

        # ✅ 2. LLM 智能分析 → 生成嵌入策略
        strategy = strategist.analyze(str(input_path))
        print(f"[🧠 LLM策略] {strategy}")

        # ✅ 3. 根据策略选择算法并传入参数
        embed_kwargs = {
            "strength": strategy["embed_strength"],
            "content_type": strategy["content_type"],
            "sensitive_regions": strategy["sensitive_regions"]
        }

        if ext == '.png':
            lsb_embed(str(input_path), str(output_path), uid, **embed_kwargs)
        else:
            dct_embed(str(input_path), str(output_path), uid, **embed_kwargs)

        # ✅ 4. 返回图片 + 策略（供前端展示决策过程）
        result_filename = f"watermarked_{Path(file.filename).stem}.png"
        result_url = f"/static/temp/{output_path.name}"

        static_temp = BASE_DIR / "static" / "temp"
        static_temp.mkdir(parents=True, exist_ok=True)
        final_static_path = static_temp / output_path.name
        output_path.rename(final_static_path)

        safe_unlink(input_path)
        
        # ✅ 记录操作日志
        log_watermark_operation(uid, file.filename, "embed", strategy=strategy["content_type"])

        return JSONResponse({
            "status": "success",
            "image_url": result_url,
            "filename": result_filename,
            "strategy": strategy
        })

    except Exception as exc:
        safe_unlink(input_path)
        safe_unlink(output_path)
        raise HTTPException(status_code=500, detail=f"水印嵌入失败: {str(exc)}") from exc


@app.post("/batch-watermark", summary="批量嵌入水印")
async def batch_add_watermark(
        files: List[UploadFile] = File(...),
        uid_prefix: str = Form("UID", description="UID前缀")
):
    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一个文件")

    results = []
    errors = []

    for idx, file in enumerate(files, 1):
        if not file.filename:
            errors.append({"index": idx, "error": "文件名不能为空"})
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ('.png', '.jpg', '.jpeg'):
            errors.append({"index": idx, "filename": file.filename, "error": "不支持的格式"})
            continue

        try:
            unique_id = str(uuid.uuid4())
            input_path = TEMP_DIR / f"batch_input_{unique_id}{ext}"
            output_path = TEMP_DIR / f"batch_watermarked_{unique_id}.png"

            content = await file.read()
            input_path.write_bytes(content)

            file_uid = f"{uid_prefix}-{idx:03d}"

            strategy = strategist.analyze(str(input_path))
            print(f"[🧠 批量处理 {idx}/{len(files)}] {file.filename} → 策略: {strategy}")

            embed_kwargs = {
                "strength": strategy["embed_strength"],
                "content_type": strategy["content_type"],
                "sensitive_regions": strategy["sensitive_regions"]
            }

            if ext == '.png':
                lsb_embed(str(input_path), str(output_path), file_uid, **embed_kwargs)
            else:
                dct_embed(str(input_path), str(output_path), file_uid, **embed_kwargs)

            static_temp = BASE_DIR / "static" / "temp"
            static_temp.mkdir(parents=True, exist_ok=True)
            final_static_path = static_temp / output_path.name
            output_path.rename(final_static_path)

            results.append({
                "index": idx,
                "filename": file.filename,
                "uid": file_uid,
                "image_url": f"/static/temp/{final_static_path.name}",
                "strategy": strategy,
                "status": "success"
            })

            safe_unlink(input_path)

        except Exception as exc:
            errors.append({
                "index": idx,
                "filename": file.filename,
                "error": str(exc)
            })

    return {
        "status": "completed",
        "total_files": len(files),
        "success_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors
    }


@app.post("/verify", summary="验证水印")
async def verify_watermark(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式")

    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"verify_{unique_id}{ext}"

    try:
        input_path.write_bytes(await file.read())

        uid = None
        for extractor in (lsb_extract, dct_extract):
            try:
                uid = extractor(str(input_path))
                if uid:
                    break
            except Exception:
                continue

        safe_unlink(input_path)
        
        # ✅ 记录操作日志
        log_watermark_operation(uid or "unknown", file.filename, "verify")

        if uid:
            return {"status": "success", "uid": uid}
        return {"status": "fail", "message": "未检测到有效水印"}

    except Exception as e:
        safe_unlink(input_path)
        return {"status": "fail", "message": str(e)}


@app.post("/detect", summary="AI防伪检测")
async def detect_tamper(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式")

    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"detect_{unique_id}{ext}"

    try:
        input_path.write_bytes(await file.read())

        # ✅ 1. AI 篡改检测
        tamper_result = detect_tampering(str(input_path))
        print(f"[🔍 AI检测结果] {tamper_result}")

        # ✅ 2. 水印验证
        uid = None
        for extractor in (lsb_extract, dct_extract):
            try:
                uid = extractor(str(input_path))
                if uid:
                    break
            except Exception:
                continue

        safe_unlink(input_path)
        
        # ✅ 记录操作日志
        log_watermark_operation(uid or "unknown", file.filename, "detect")

        # ✅ 3. 生成综合报告
        report = {
            "status": "success",
            "tamper_detection": tamper_result,
            "watermark_verified": uid is not None,
            "uid": uid,
            "integrity_score": _calculate_integrity_score(tamper_result, uid is not None)
        }

        return report

    except Exception as e:
        safe_unlink(input_path)
        return {"status": "fail", "message": str(e)}


def _calculate_integrity_score(tamper_result: dict, watermark_valid: bool) -> float:
    """计算综合完整性评分"""
    score = 1.0
    
    if tamper_result["is_tampered"]:
        score -= tamper_result["confidence"] * 0.5
    
    if not watermark_valid:
        score -= 0.3
    
    return max(0.0, min(1.0, score))


@app.get("/strength-levels", summary="获取强度级别列表")
def get_strength_levels():
    return {
        "status": "success",
        "levels": STRENGTH_LEVELS
    }


@app.post("/analyze-strength", summary="分析图像并建议水印强度")
async def analyze_image_strength(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式")

    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"analyze_{unique_id}{ext}"

    try:
        input_path.write_bytes(await file.read())

        analysis_result = calculate_adaptive_strength(str(input_path))

        safe_unlink(input_path)

        return JSONResponse({
            "status": "success",
            "analysis": analysis_result,
            "strength_levels": STRENGTH_LEVELS
        })

    except Exception as e:
        safe_unlink(input_path)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}") from e


@app.post("/adaptive-watermark", summary="自适应强度嵌入水印")
async def add_adaptive_watermark(
        file: UploadFile = File(...),
        uid: str = Form(..., description="水印UID"),
        auto_strength: bool = Form(False, description="是否启用自适应强度")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式")

    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"adaptive_{unique_id}{ext}"
    output_path = TEMP_DIR / f"adaptive_watermarked_{unique_id}.png"

    try:
        input_path.write_bytes(await file.read())

        if auto_strength:
            result = embed_adaptive_watermark(str(input_path), str(output_path), uid)
        else:
            if ext == '.png':
                lsb_embed(str(input_path), str(output_path), uid)
            else:
                dct_embed(str(input_path), str(output_path), uid)
            result = {"status": "success", "uid": uid}

        static_temp = BASE_DIR / "static" / "temp"
        static_temp.mkdir(parents=True, exist_ok=True)
        output_filename = f"adaptive_watermarked_{unique_id}.png"
        static_output = static_temp / output_filename
        output_path.rename(static_output)

        safe_unlink(input_path)
        
        # ✅ 记录操作日志
        log_watermark_operation(uid, file.filename, "embed", strategy="adaptive")

        return JSONResponse({
            **result,
            "watermarked_url": f"/static/temp/{output_filename}"
        })

    except Exception as e:
        safe_unlink(input_path)
        safe_unlink(output_path)
        raise HTTPException(status_code=500, detail=f"嵌入失败: {str(e)}") from e


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TraceMark v2.4", "llm_enabled": True, "tamper_detection": True, "adaptive_strength": True}


@app.get("/watermark-history", summary="查询水印操作历史")
async def get_watermark_history_api(uid: str = None, limit: int = 20):
    history = get_watermark_history(uid, limit)
    return {"status": "success", "data": history}


@app.get("/watermark-stats", summary="获取水印统计数据")
async def get_watermark_stats_api(uid: str = None):
    stats = get_watermark_stats(uid)
    return {"status": "success", "stats": stats}


@app.delete("/clear-history", summary="清除操作历史")
async def clear_history_api(uid: str = None):
    try:
        deleted_count = clear_watermark_history(uid)
        if uid:
            message = f"已清除 UID '{uid}' 的 {deleted_count} 条操作记录"
        else:
            message = f"已清除所有 {deleted_count} 条操作记录"
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TraceMark v2.4 (AI防伪检测版) - 启动成功！")
    print("="*60)
    print("\n📱 前端页面 (点击访问):")
    print("   http://localhost:8000")
    print("\n📚 API 文档 (点击访问):")
    print("   http://localhost:8000/docs")
    print("\n✨ 新增功能:")
    print("   - AI 防伪检测")
    print("   - LLM 智能水印策略")
    print("   - 综合完整性评分")
    print("   - 水印强度自适应")
    print("\n" + "="*60)
    print("按 Ctrl+C 停止服务")
    print("="*60 + "\n")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")