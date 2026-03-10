"""
PhotoRestore API — v2.1
Pipeline:
  /restore      → CodeFormer 2x (face + hair + background) + Real-ESRGAN 2x (full image) = 4x total
  /portrait     → CodeFormer 4x only (faster, identity-preserving, best for selfies)
  /enhance-only → Real-ESRGAN 4x only (landscapes, full body, no-face photos)
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
    description="Full-image AI restoration: face, hair, background, full body, upscale",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Model version hashes ────────────────────────────────────────────
CODEFORMER  = "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2"
REAL_ESRGAN = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"


# ── Shared helpers ──────────────────────────────────────────────────

def _make_data_uri(contents: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(contents).decode()}"


async def _fetch_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0) as client:
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


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "version": "2.1.0", "message": "PhotoRestore API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    fidelity: float = Form(default=0.5)
):
    """
    FULL RESTORE — Best for old/damaged/pixelated photos with faces.

    Pipeline:
      Step 1 → CodeFormer 2x : restores face, hair strands, skin, background
      Step 2 → Real-ESRGAN 2x: sharpens full image (body, clothes, background)
      Total  → 4x upscale

    fidelity (0.0 to 1.0):
      0.2 = very old / heavily damaged photos
      0.5 = balanced — good for most photos  (default)
      0.7 = modern photos — preserve identity
    """
    contents = await file.read()
    _validate(file, contents)
    fidelity = max(0.0, min(1.0, fidelity))
    data_uri = _make_data_uri(contents, file.content_type or "image/jpeg")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    try:
        # ── Step 1: CodeFormer ──────────────────────────────────────
        print(f"[INFO] /restore — CodeFormer fidelity={fidelity} file={file.filename}")
        cf_out = client.run(
            CODEFORMER,
            input={
                "image":               data_uri,
                "codeformer_fidelity": fidelity,
                "background_enhance":  True,   # sharpen background
                "face_upsample":       True,   # recover hair + skin detail
                "upscale":             2,      # 2x in this pass
            }
        )
        cf_bytes = await _fetch_url(str(cf_out))
        print(f"[INFO] CodeFormer done — {len(cf_bytes):,} bytes")

        # ── Step 2: Real-ESRGAN ─────────────────────────────────────
        print("[INFO] Real-ESRGAN full image upscale...")
        esrgan_out = client.run(
            REAL_ESRGAN,
            input={
                "image":        _make_data_uri(cf_bytes, "image/png"),
                "scale":        2,       # another 2x = 4x total
                "face_enhance": True,    # extra face/hair pass
            }
        )
        final_bytes = await _fetch_url(str(esrgan_out))
        print(f"[INFO] Done — {len(final_bytes):,} bytes (4x total)")

        return JSONResponse({
            "success":             True,
            "image_base64":        base64.b64encode(final_bytes).decode(),
            "mime_type":           "image/png",
            "pipeline":            "codeformer-2x + real-esrgan-2x = 4x",
            "original_size_bytes": len(contents),
            "restored_size_bytes": len(final_bytes),
            "upscale_factor":      4,
        })

    except replicate.exceptions.ReplicateError as e:
        print(f"[ERROR] Replicate: {e}")
        raise HTTPException(502, f"AI processing failed: {e}")
    except httpx.HTTPError as e:
        print(f"[ERROR] HTTP: {e}")
        raise HTTPException(502, "Failed to retrieve image from AI service.")
    except Exception as e:
        print(f"[ERROR] Unexpected: {e}")
        raise HTTPException(500, f"Internal error: {e}")


@app.post("/portrait")
async def restore_portrait(
    file: UploadFile = File(...),
    fidelity: float = Form(default=0.7)
):
    """
    PORTRAIT MODE — Best for modern selfies / close-up portraits.
    Faster than Full Restore (single pass).
    Higher default fidelity (0.7) preserves face identity.

    Pipeline:
      Step 1 → CodeFormer 4x only: face + hair + skin + background in one pass
    """
    contents = await file.read()
    _validate(file, contents)
    fidelity = max(0.0, min(1.0, fidelity))
    data_uri = _make_data_uri(contents, file.content_type or "image/jpeg")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    try:
        print(f"[INFO] /portrait — CodeFormer fidelity={fidelity} file={file.filename}")
        out = client.run(
            CODEFORMER,
            input={
                "image":               data_uri,
                "codeformer_fidelity": fidelity,
                "background_enhance":  True,
                "face_upsample":       True,
                "upscale":             4,   # 4x in one pass — fast
            }
        )
        final_bytes = await _fetch_url(str(out))
        print(f"[INFO] Portrait done — {len(final_bytes):,} bytes")

        return JSONResponse({
            "success":             True,
            "image_base64":        base64.b64encode(final_bytes).decode(),
            "mime_type":           "image/png",
            "pipeline":            "codeformer-portrait-4x",
            "original_size_bytes": len(contents),
            "restored_size_bytes": len(final_bytes),
            "upscale_factor":      4,
        })

    except replicate.exceptions.ReplicateError as e:
        print(f"[ERROR] Replicate: {e}")
        raise HTTPException(502, f"AI processing failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected: {e}")
        raise HTTPException(500, f"Internal error: {e}")


@app.post("/enhance-only")
async def enhance_only(file: UploadFile = File(...)):
    """
    ENHANCE ONLY — Best for full body photos, landscapes, animals, objects.
    No face-specific processing — sharpens the ENTIRE image equally.

    Pipeline:
      Step 1 → Real-ESRGAN 4x: whole image super-resolution
    """
    contents = await file.read()
    _validate(file, contents)
    data_uri = _make_data_uri(contents, file.content_type or "image/jpeg")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    try:
        print(f"[INFO] /enhance-only — Real-ESRGAN 4x file={file.filename}")
        out = client.run(
            REAL_ESRGAN,
            input={
                "image":        data_uri,
                "scale":        4,
                "face_enhance": False,  # pure image upscale, no face bias
            }
        )
        final_bytes = await _fetch_url(str(out))
        print(f"[INFO] Enhance done — {len(final_bytes):,} bytes")

        return JSONResponse({
            "success":        True,
            "image_base64":   base64.b64encode(final_bytes).decode(),
            "mime_type":      "image/png",
            "pipeline":       "real-esrgan-4x",
            "upscale_factor": 4,
        })

    except replicate.exceptions.ReplicateError as e:
        print(f"[ERROR] Replicate: {e}")
        raise HTTPException(502, f"AI processing failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
