"""
PhotoRestore API — v2.0
Pipeline:
  /restore        → CodeFormer (face + hair + background) + Real-ESRGAN final upscale
  /portrait       → CodeFormer high-fidelity (preserves identity, fixes hair/skin, faster)
  /enhance-only   → Real-ESRGAN 4x (landscapes / no-face photos)
"""

import os
import base64
import httpx
import replicate
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="PhotoRestore API",
    description="Full-image AI restoration: face, hair, background, upscale",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# ─────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────

def _make_data_uri(contents: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(contents).decode()}"


async def _fetch_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def _validate(file: UploadFile, contents: bytes):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(400, "Invalid file type. Use JPEG, PNG, or WEBP.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File too large. Max 10 MB.")
    if not REPLICATE_API_TOKEN:
        raise HTTPException(500, "REPLICATE_API_TOKEN not set on server.")


# ─────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "version": "2.0.0", "message": "PhotoRestore API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    fidelity: float = Form(default=0.5)
):
    """
    PRIMARY endpoint — full photo restoration (face + hair + background + upscale).

    Pipeline:
      1. CodeFormer  — recovers face detail, hair strands, skin texture,
                       AND enhances background (not just faces).
      2. Real-ESRGAN — final 2x super-resolution across the whole image.
      Total: 4x upscale from original.

    fidelity (0.0–1.0):
      0.0 = max restoration  (heavily degraded / very old photos)
      0.5 = balanced         (default — works for most photos)
      1.0 = max fidelity     (keep original look, just denoise/sharpen)
    """
    contents = await file.read()
    _validate(file, contents)
    fidelity = max(0.0, min(1.0, fidelity))
    data_uri = _make_data_uri(contents, file.content_type or "image/jpeg")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    try:
        # ── STEP 1: CodeFormer ──────────────────────────────────────────
        # background_enhance=True  → sharpens blurry/noisy backgrounds
        # face_upsample=True       → recovers fine hair strands + skin pores
        # upscale=2                → 2x upscale in this pass
        print(f"[INFO] CodeFormer — fidelity={fidelity} file={file.filename}")
        cf_out = client.run(
            "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
            input={
                "image":               data_uri,
                "codeformer_fidelity": fidelity,
                "background_enhance":  True,
                "face_upsample":       True,
                "upscale":             2,
            }
        )
        cf_bytes = await _fetch_url(str(cf_out))
        print(f"[INFO] CodeFormer done — {len(cf_bytes):,} bytes")

        # ── STEP 2: Real-ESRGAN final pass ──────────────────────────────
        # Adds another 2x sharpening + upscale across the whole image.
        # face_enhance=True here adds a third pass of hair/face refinement.
        print("[INFO] Real-ESRGAN final upscale pass...")
        esrgan_out = client.run(
            "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee2d96b1a3a1ddbbe7751",
            input={
                "image":        _make_data_uri(cf_bytes, "image/png"),
                "scale":        2,
                "face_enhance": True,
            }
        )
        final_bytes = await _fetch_url(str(esrgan_out))
        print(f"[INFO] Pipeline done — final {len(final_bytes):,} bytes (4x upscale)")

        return JSONResponse({
            "success":             True,
            "image_base64":        base64.b64encode(final_bytes).decode(),
            "mime_type":           "image/png",
            "pipeline":            "codeformer + real-esrgan",
            "original_size_bytes": len(contents),
            "restored_size_bytes": len(final_bytes),
            "upscale_factor":      4,
        })

    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(502, f"AI processing failed: {e}")
    except httpx.HTTPError as e:
        raise HTTPException(502, "Failed to retrieve image from AI service.")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@app.post("/portrait")
async def restore_portrait(
    file: UploadFile = File(...),
    fidelity: float = Form(default=0.7)
):
    """
    PORTRAIT mode — faster, identity-preserving single CodeFormer pass.
    Best for: modern selfies / portraits that need hair + skin detail sharpening.
    Higher default fidelity (0.7) keeps the person's face recognisable.
    """
    contents = await file.read()
    _validate(file, contents)
    fidelity = max(0.0, min(1.0, fidelity))
    data_uri = _make_data_uri(contents, file.content_type or "image/jpeg")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    try:
        print(f"[INFO] Portrait mode — fidelity={fidelity}")
        out = client.run(
            "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142ebb9d5ba4f",
            input={
                "image":               data_uri,
                "codeformer_fidelity": fidelity,
                "background_enhance":  True,
                "face_upsample":       True,
                "upscale":             4,   # 4x in one pass — fast for portraits
            }
        )
        final_bytes = await _fetch_url(str(out))
        return JSONResponse({
            "success":             True,
            "image_base64":        base64.b64encode(final_bytes).decode(),
            "mime_type":           "image/png",
            "pipeline":            "codeformer-portrait",
            "original_size_bytes": len(contents),
            "restored_size_bytes": len(final_bytes),
            "upscale_factor":      4,
        })

    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(502, f"AI processing failed: {e}")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@app.post("/enhance-only")
async def enhance_only(file: UploadFile = File(...)):
    """
    ENHANCE ONLY — Real-ESRGAN 4x.
    Best for: landscapes, animals, objects, architecture (no faces).
    """
    contents = await file.read()
    _validate(file, contents)
    data_uri = _make_data_uri(contents, file.content_type or "image/jpeg")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    try:
        out = client.run(
            "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee2d96b1a3a1ddbbe7751",
            input={"image": data_uri, "scale": 4, "face_enhance": False}
        )
        final_bytes = await _fetch_url(str(out))
        return JSONResponse({
            "success":        True,
            "image_base64":   base64.b64encode(final_bytes).decode(),
            "mime_type":      "image/png",
            "pipeline":       "real-esrgan-4x",
            "upscale_factor": 4,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
