"""
PhotoRestore API — v3.0  (FREE — No API key needed)
Uses FREE public HuggingFace Spaces via gradio_client:
  /restore      → GFPGAN v1.4 (face + hair + background) — FREE
  /portrait     → GFPGAN v1.4 higher fidelity — FREE
  /enhance-only → GFPGAN without face-specific pass — FREE
"""

import os
import base64
import httpx
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
import uvicorn

app = FastAPI(
    title="PhotoRestore API",
    description="Free AI photo restoration via HuggingFace Spaces",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Free HuggingFace Space — GFPGAN by original authors (Xintao/Tencent ARC)
HF_SPACE_GFPGAN = "Xintao/GFPGAN"


# ── Shared helpers ──────────────────────────────────────────────────

def _validate(file: UploadFile, contents: bytes):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(400, "Invalid file type. Use JPEG, PNG, or WEBP.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File too large. Max 10 MB.")


def _bytes_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _save_upload_to_temp(contents: bytes, suffix: str = ".jpg") -> str:
    """Save uploaded bytes to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(contents)
    tmp.close()
    return tmp.name


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "version": "3.0.0", "message": "PhotoRestore API running (FREE via HuggingFace)"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    fidelity: float = Form(default=0.5)
):
    """
    FULL RESTORE — Face + hair + background using GFPGAN (FREE).
    Best for old/damaged/pixelated photos.
    """
    contents = await file.read()
    _validate(file, contents)

    # Determine file extension
    ext = ".jpg"
    if file.content_type == "image/png":
        ext = ".png"
    elif file.content_type == "image/webp":
        ext = ".webp"

    tmp_input = _save_upload_to_temp(contents, suffix=ext)

    try:
        print(f"[INFO] /restore — GFPGAN via HuggingFace Space file={file.filename}")
        client = Client(HF_SPACE_GFPGAN)

        # Call the GFPGAN Space
        # Parameters: img, version, scale
        result = client.predict(
            img=handle_file(tmp_input),
            version="v1.4",      # best version — face + hair + background
            scale=2,             # 2x upscale
            api_name="/predict"
        )

        # Result is a tuple: (restored_img_path, original_img_path)
        # or just a path string depending on Space version
        if isinstance(result, (list, tuple)):
            output_path = result[0]
        else:
            output_path = result

        result_b64 = _bytes_to_base64(output_path)
        print(f"[INFO] Restore done — output: {output_path}")

        return JSONResponse({
            "success":        True,
            "image_base64":   result_b64,
            "mime_type":      "image/png",
            "pipeline":       "gfpgan-v1.4-free",
            "upscale_factor": 2,
        })

    except Exception as e:
        print(f"[ERROR] /restore failed: {e}")
        raise HTTPException(502, f"AI processing failed: {str(e)}")
    finally:
        Path(tmp_input).unlink(missing_ok=True)


@app.post("/portrait")
async def restore_portrait(
    file: UploadFile = File(...),
    fidelity: float = Form(default=0.7)
):
    """
    PORTRAIT MODE — GFPGAN 4x upscale (FREE).
    Best for modern selfies / close-up portraits.
    """
    contents = await file.read()
    _validate(file, contents)

    ext = ".jpg"
    if file.content_type == "image/png":
        ext = ".png"

    tmp_input = _save_upload_to_temp(contents, suffix=ext)

    try:
        print(f"[INFO] /portrait — GFPGAN 4x file={file.filename}")
        client = Client(HF_SPACE_GFPGAN)

        result = client.predict(
            img=handle_file(tmp_input),
            version="v1.4",
            scale=4,             # 4x upscale for portrait
            api_name="/predict"
        )

        if isinstance(result, (list, tuple)):
            output_path = result[0]
        else:
            output_path = result

        result_b64 = _bytes_to_base64(output_path)
        print(f"[INFO] Portrait done")

        return JSONResponse({
            "success":        True,
            "image_base64":   result_b64,
            "mime_type":      "image/png",
            "pipeline":       "gfpgan-v1.4-portrait-4x-free",
            "upscale_factor": 4,
        })

    except Exception as e:
        print(f"[ERROR] /portrait failed: {e}")
        raise HTTPException(502, f"AI processing failed: {str(e)}")
    finally:
        Path(tmp_input).unlink(missing_ok=True)


@app.post("/enhance-only")
async def enhance_only(file: UploadFile = File(...)):
    """
    ENHANCE ONLY — GFPGAN 4x for full body / landscapes (FREE).
    Uses RestoreFormer for more natural enhancement without face hallucination.
    """
    contents = await file.read()
    _validate(file, contents)

    ext = ".jpg"
    if file.content_type == "image/png":
        ext = ".png"

    tmp_input = _save_upload_to_temp(contents, suffix=ext)

    try:
        print(f"[INFO] /enhance-only — GFPGAN RestoreFormer file={file.filename}")
        client = Client(HF_SPACE_GFPGAN)

        result = client.predict(
            img=handle_file(tmp_input),
            version="RestoreFormer",  # better for full body / non-face focus
            scale=4,
            api_name="/predict"
        )

        if isinstance(result, (list, tuple)):
            output_path = result[0]
        else:
            output_path = result

        result_b64 = _bytes_to_base64(output_path)
        print(f"[INFO] Enhance done")

        return JSONResponse({
            "success":        True,
            "image_base64":   result_b64,
            "mime_type":      "image/png",
            "pipeline":       "restoreformer-4x-free",
            "upscale_factor": 4,
        })

    except Exception as e:
        print(f"[ERROR] /enhance-only failed: {e}")
        raise HTTPException(502, f"AI processing failed: {str(e)}")
    finally:
        Path(tmp_input).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
