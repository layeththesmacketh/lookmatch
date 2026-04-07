#!/usr/bin/env python3
"""LookMatch — Photo Color Matching Tool for Photographers"""

import os
import io
import uuid
import json
import shutil
import zipfile
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rawpy
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
from skimage.exposure import match_histograms
import tifffile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import uvicorn

# --- Config ---
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PREVIEW_DIR = BASE_DIR / "previews"
EXPORT_DIR = BASE_DIR / "exports"

for d in [UPLOAD_DIR, PREVIEW_DIR, EXPORT_DIR]:
    d.mkdir(exist_ok=True)

PREVIEW_MAX_EDGE = 1600
RAW_EXTENSIONS = {
    ".cr3", ".cr2", ".crw",          # Canon
    ".nef", ".nrw",                   # Nikon
    ".arw", ".srf", ".sr2",          # Sony
    ".dng",                           # Adobe / Apple ProRAW / Leica / Hasselblad
    ".orf",                           # Olympus / OM System
    ".raf",                           # Fujifilm
    ".rw2", ".rwl",                   # Panasonic / Leica
    ".pef", ".ptx",                   # Pentax
    ".3fr",                           # Hasselblad
    ".srw",                           # Samsung
    ".mrw",                           # Minolta
    ".dcr", ".kdc",                   # Kodak
    ".x3f",                           # Sigma
    ".erf",                           # Epson
    ".bay",                           # Casio
    ".mos",                           # Leaf
    ".iiq",                           # Phase One
}
HEIF_EXTENSIONS = {".heic", ".heif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".avif", ".psd", ".gif"}
ALL_EXTENSIONS = RAW_EXTENSIONS | HEIF_EXTENSIONS | IMAGE_EXTENSIONS

app = FastAPI(title="LookMatch")

# --- Optional password protection ---
# Set LOOKMATCH_PASSWORD env var to enable HTTP Basic Auth on all routes
_PASSWORD = os.environ.get("LOOKMATCH_PASSWORD", "")

@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    if not _PASSWORD or request.url.path == "/health":
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            _, pwd = decoded.split(":", 1)
            if secrets.compare_digest(pwd.encode(), _PASSWORD.encode()):
                return await call_next(request)
        except Exception:
            pass
    from fastapi.responses import Response as _Resp
    return _Resp(
        status_code=401,
        headers={"WWW-Authenticate": "Basic realm='LookMatch'"},
    )

def check_auth():
    pass  # kept for compatibility

# In-memory session state
session = {
    "reference": None,       # {"id": ..., "filename": ..., "path": ..., "preview": ...}
    "targets": [],           # list of same structure
    "matched": {},           # target_id -> {"preview": path, "full_path": path}
    "settings": {
        "method": "reinhard",
        "strength": 85,
        "film_stock": None,
        "film_strength": 100,
        "film_overrides": {},
        "adjustments": {
            "temp": 0, "tint": 0,
            "exposure": 0.0, "contrast": 0,
            "highlights": 0, "shadows": 0,
            "whites": 0, "blacks": 0,
            "vibrance": 0, "saturation": 0,
            "clarity": 0, "texture": 0,
        },
    }
}


# --- Image I/O Helpers ---

def read_image(filepath: str, half_size: bool = False) -> np.ndarray:
    """Read any supported image file, return BGR numpy array (16-bit for RAW, 8-bit for standard)."""
    ext = Path(filepath).suffix.lower()

    if ext in RAW_EXTENSIONS:
        try:
            with rawpy.imread(filepath) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=half_size,
                    no_auto_bright=True,
                    output_bps=16,
                    output_color=rawpy.ColorSpace.sRGB,
                )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as e:
            raise ValueError(f"Failed to decode RAW file ({ext}): {e}")

    if ext in HEIF_EXTENSIONS:
        try:
            pil_img = Image.open(filepath)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to decode HEIC/HEIF file: {e}")

    # Standard image formats (JPEG, PNG, TIFF, WebP, AVIF, etc.)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback to Pillow (handles AVIF, PSD, GIF, and edge cases cv2 misses)
        try:
            pil_img = Image.open(filepath)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Cannot read image ({ext}). File may be corrupted or unsupported.")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_preview(img: np.ndarray, max_edge: int = PREVIEW_MAX_EDGE) -> np.ndarray:
    """Downscale for preview, convert to 8-bit."""
    # Convert to 8-bit if needed
    if img.dtype == np.uint16:
        img8 = (img / 256).astype(np.uint8)
    else:
        img8 = img.copy()

    h, w = img8.shape[:2]
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        img8 = cv2.resize(img8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img8


def save_preview(img: np.ndarray, file_id: str) -> str:
    """Save preview as JPEG, return filename."""
    preview_name = f"{file_id}_preview.jpg"
    preview_path = PREVIEW_DIR / preview_name
    cv2.imwrite(str(preview_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return preview_name


# --- Color Matching ---

def reinhard_transfer(source: np.ndarray, reference: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Reinhard color transfer in LAB space. Works on 8-bit BGR images."""
    src_f = source.astype(np.float32)
    ref_f = reference.astype(np.float32)

    src_lab = cv2.cvtColor(src_f / 255.0, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref_f / 255.0, cv2.COLOR_BGR2LAB)

    result_lab = np.copy(src_lab)
    for ch in range(3):
        src_mean, src_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std()
        ref_mean, ref_std = ref_lab[:, :, ch].mean(), ref_lab[:, :, ch].std()

        if src_std < 1e-6:
            src_std = 1e-6

        transferred = (src_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean
        # Blend with original based on strength
        result_lab[:, :, ch] = src_lab[:, :, ch] * (1 - strength) + transferred * strength

    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR) * 255.0
    result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    return result_bgr


def histogram_match(source: np.ndarray, reference: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Histogram matching using scikit-image. 8-bit BGR images."""
    # skimage expects RGB
    src_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

    matched_rgb = match_histograms(src_rgb, ref_rgb, channel_axis=-1).astype(np.uint8)

    if strength < 1.0:
        matched_rgb = (src_rgb.astype(np.float32) * (1 - strength) +
                       matched_rgb.astype(np.float32) * strength)
        matched_rgb = np.clip(matched_rgb, 0, 255).astype(np.uint8)

    return cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR)


def apply_color_match(source_bgr: np.ndarray, reference_bgr: np.ndarray,
                      method: str = "reinhard", strength: float = 0.85) -> np.ndarray:
    """Apply color matching. Handles any bit depth by converting to 8-bit for matching,
    then applying the transformation to the original."""
    # Convert to 8-bit for matching algorithm
    if source_bgr.dtype == np.uint16:
        src8 = (source_bgr / 256).astype(np.uint8)
    else:
        src8 = source_bgr.copy()

    if reference_bgr.dtype == np.uint16:
        ref8 = (reference_bgr / 256).astype(np.uint8)
    else:
        ref8 = reference_bgr.copy()

    if method == "reinhard":
        matched8 = reinhard_transfer(src8, ref8, strength)
    else:
        matched8 = histogram_match(src8, ref8, strength)

    # If source was 16-bit, scale the matched result back up
    if source_bgr.dtype == np.uint16:
        # Compute the per-pixel ratio of transformation and apply to 16-bit
        src8_f = src8.astype(np.float32) + 1e-6
        matched_f = matched8.astype(np.float32)
        ratio = matched_f / src8_f
        result16 = (source_bgr.astype(np.float32) * ratio)
        result16 = np.clip(result16, 0, 65535).astype(np.uint16)
        return result16
    else:
        return matched8


# --- Film Stock Definitions ---

FILM_STOCKS = {
    "portra400": {
        "name": "Kodak Portra 400",
        "description": "Warm skin tones, creamy highlights, fine grain",
        "shadow_lift_bgr": [2, 3, 10],
        "highlight_push_bgr": [0, 1, 4],
        "saturation": 0.85,
        "contrast": 1.02,
        "grain_amount": 12,        # RMS 10; T-grain — uniform, flat spatial spectrum
        "grain_size": 1.1,
        "grain_roughness": 0.0,    # T-grain: pure Gaussian, no clustering
        "fade": 7,
        "halation": 0,
        "bw": False,
    },
    "ektar100": {
        "name": "Kodak Ektar 100",
        "description": "Ultra-saturated, punchy colors, near-zero grain",
        "shadow_lift_bgr": [3, 1, 1],
        "highlight_push_bgr": [0, 0, 3],
        "saturation": 1.35,
        "contrast": 1.12,
        "grain_amount": 9,         # RMS 8; T-grain — finest grain in lineup
        "grain_size": 0.7,
        "grain_roughness": 0.0,    # T-grain
        "fade": 2,
        "halation": 0,
        "bw": False,
    },
    "superia400": {
        "name": "Fujifilm Superia 400",
        "description": "Cooler tones with green-tinted midtones",
        "shadow_lift_bgr": [5, 4, 0],
        "highlight_push_bgr": [2, 2, 0],
        "midtone_green": 6,
        "saturation": 0.95,
        "contrast": 1.06,
        "grain_amount": 14,        # RMS ~11-12; cubic-crystal
        "grain_size": 1.3,
        "grain_roughness": 0.5,    # Cubic: moderate organic clustering
        "fade": 4,
        "halation": 0,
        "bw": False,
    },
    "velvia50": {
        "name": "Fujifilm Velvia 50",
        "description": "Hyper-vivid slide film with deep blacks",
        "shadow_lift_bgr": [3, 0, 1],
        "highlight_push_bgr": [0, 1, 2],
        "saturation": 1.55,
        "contrast": 1.22,
        "grain_amount": 9,         # RMS 8; slide film — very fine, minimal clustering
        "grain_size": 0.7,
        "grain_roughness": 0.1,    # Near T-grain character
        "fade": 0,
        "halation": 0,
        "bw": False,
    },
    "gold200": {
        "name": "Kodak Gold 200",
        "description": "Warm golden tones, classic consumer film",
        "shadow_lift_bgr": [2, 5, 14],
        "highlight_push_bgr": [0, 4, 14],
        "saturation": 1.05,
        "contrast": 1.08,
        "grain_amount": 14,        # RMS 11; cubic-crystal — organic, slightly clumpy
        "grain_size": 1.5,
        "grain_roughness": 0.6,    # Cubic: visible mid-frequency clusters
        "fade": 6,
        "halation": 0,
        "bw": False,
    },
    "cinestill800t": {
        "name": "CineStill 800T",
        "description": "Cinematic tungsten film with red halation",
        "shadow_lift_bgr": [10, 5, 2],
        "highlight_push_bgr": [2, 1, 6],
        "saturation": 0.90,
        "contrast": 1.10,
        "grain_amount": 18,        # RMS ~13-15 at EI800 (Vision3 500T pushed)
        "grain_size": 1.7,
        "grain_roughness": 0.4,    # Cubic (cinema stock base)
        "fade": 12,
        "halation": 18,            # sigma at 2000px ref width; scales with image size
        "bw": False,
    },
    "hp5": {
        "name": "Ilford HP5 Plus",
        "description": "Classic B&W, versatile with rich midtones",
        "shadow_lift_bgr": [0, 0, 0],
        "highlight_push_bgr": [0, 0, 0],
        "saturation": 0.0,
        "contrast": 1.05,
        "grain_amount": 15,        # RMS 12; cubic-crystal — most organic, clustered grain
        "grain_size": 1.4,
        "grain_roughness": 0.7,    # Cubic: most clustered / organic in lineup
        "fade": 4,
        "halation": 0,
        "bw": True,
        "bw_warmth": 0.03,
    },
    "tmax400": {
        "name": "Kodak T-Max 400",
        "description": "B&W with fine grain and clinical sharpness",
        "shadow_lift_bgr": [0, 0, 0],
        "highlight_push_bgr": [0, 0, 0],
        "saturation": 0.0,
        "contrast": 1.17,
        "grain_amount": 11,        # RMS 10; T-grain — fine, uniform, no clustering
        "grain_size": 0.9,
        "grain_roughness": 0.0,    # T-grain: pure Gaussian, clinical uniformity
        "fade": 2,
        "halation": 0,
        "bw": True,
        "bw_warmth": 0.0,
    },
}


def apply_film_stock(img_bgr: np.ndarray, stock_id: str, strength: float = 1.0,
                     skip_grain: bool = False) -> np.ndarray:
    """Apply film stock emulation to an 8-bit BGR image."""
    if stock_id not in FILM_STOCKS:
        raise ValueError(f"Unknown film stock: {stock_id}")

    p = FILM_STOCKS[stock_id]
    original = img_bgr.copy()
    img = img_bgr.astype(np.float32)

    # Luminance for masking
    lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    # 1. Fade — lift blacks (simulates film base fog)
    fade = p.get("fade", 0)
    if fade:
        img = img * ((255.0 - fade) / 255.0) + fade

    # 2. Contrast (simple linear stretch around 128)
    contrast = p.get("contrast", 1.0)
    if contrast != 1.0:
        img = (img - 128.0) * contrast + 128.0

    # 3. Split toning — shadow/highlight color grading
    shadow_mask = np.clip(1.0 - lum / 128.0, 0, 1)[:, :, np.newaxis]
    highlight_mask = np.clip((lum - 128.0) / 127.0, 0, 1)[:, :, np.newaxis]
    midtone_mask = np.clip(1.0 - np.abs(lum - 128.0) / 128.0, 0, 1)[:, :, np.newaxis]

    shadow_lift = np.array(p.get("shadow_lift_bgr", [0, 0, 0]), dtype=np.float32)
    highlight_push = np.array(p.get("highlight_push_bgr", [0, 0, 0]), dtype=np.float32)
    img = img + shadow_mask * shadow_lift + highlight_mask * highlight_push

    # Midtone green push (Fujifilm Superia characteristic)
    if p.get("midtone_green", 0):
        mg = p["midtone_green"]
        midtone_push = np.array([mg * 0.3, mg, 0], dtype=np.float32)  # BGR
        img = img + midtone_mask * midtone_push

    # 4. Saturation
    sat = p.get("saturation", 1.0)
    if sat != 1.0:
        lum_cur = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        lum3 = np.stack([lum_cur, lum_cur, lum_cur], axis=-1)
        img = lum3 + (img - lum3) * sat

    # 5. B&W conversion
    if p.get("bw"):
        lum_bw = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        warmth = p.get("bw_warmth", 0.0)
        img = np.stack([
            lum_bw * (1.0 - warmth),
            lum_bw,
            lum_bw * (1.0 + warmth),
        ], axis=-1)

    img = np.clip(img, 0, 255)

    # 6. Halation — red-orange bloom around very bright highlights (rem-jet removal artifact)
    #    Source: CineStill technical data + published emulation research
    #    - Threshold: ~90-95% luminance (tight — only specular highlights/point sources)
    #    - Bloom color: RGB (255, 80, 40) i.e. strong red, partial orange
    #    - Sigma scales linearly with image width (ref: sigma=18 at 2000px wide)
    #    - Slight horizontal elongation (1.2:1 H:V, film plane orientation)
    #    Skip for LUT generation (spatial, not per-pixel)
    if p.get("halation", 0) and not skip_grain:
        ref_sigma = float(p["halation"])
        h_img, w_img = img.shape[:2]
        # Scale sigma with image width relative to 2000px reference
        sigma_scaled = ref_sigma * (w_img / 2000.0)
        sigma_h = max(1.0, sigma_scaled * 1.2)   # wider horizontally (film plane)
        sigma_v = max(1.0, sigma_scaled * 1.0)

        lum_h = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        # Extract ONLY the highlight region (>~85% lum), then blur it to create bloom
        # Threshold at 215 to catch more highlights than 230 did; soft ramp over 40 units
        h_mask = np.clip((lum_h - 215.0) / 40.0, 0, 1)
        highlight_layer = lum_h * h_mask
        # Blur the isolated highlights — this is what makes the bloom SPREAD to dark areas
        bloom = cv2.GaussianBlur(highlight_layer, (0, 0), sigma_h, sigma_v)
        # Bloom RGB ~(255, 80, 40) → BGR (40, 80, 255): R=65%, G=20%, B=8%
        img[:, :, 2] = np.clip(img[:, :, 2] + bloom * 0.65, 0, 255)  # R
        img[:, :, 1] = np.clip(img[:, :, 1] + bloom * 0.20, 0, 255)  # G
        img[:, :, 0] = np.clip(img[:, :, 0] + bloom * 0.08, 0, 255)  # B

    # 7. Grain (skip for LUT generation)
    #    grain_roughness=0: T-grain (flat Gaussian, uniform) — Portra, Ektar, T-Max, Velvia
    #    grain_roughness>0: cubic-crystal (Gaussian + mid-freq clusters) — HP5, Gold, Superia
    grain_amount = p.get("grain_amount", 0)
    if grain_amount > 0 and not skip_grain:
        h, w = img.shape[:2]
        grain_size = p.get("grain_size", 1.0)
        roughness = p.get("grain_roughness", 0.0)

        if grain_size > 1.0:
            gh = max(1, int(h / grain_size))
            gw = max(1, int(w / grain_size))
            base_grain = np.random.normal(0, grain_amount, (gh, gw)).astype(np.float32)
            base_grain = cv2.resize(base_grain, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            base_grain = np.random.normal(0, grain_amount, (h, w)).astype(np.float32)

        if roughness > 0:
            # Cubic-crystal: add a blurred mid-frequency "clump" layer
            cluster_sigma = max(3.0, w / 250.0)   # ~4px at 1000px wide
            clump = np.random.normal(0, grain_amount * roughness, (h, w)).astype(np.float32)
            clump = cv2.GaussianBlur(clump, (0, 0), cluster_sigma)
            grain = base_grain * (1.0 - roughness * 0.4) + clump * 0.8
        else:
            grain = base_grain  # T-grain: pure Gaussian

        lum_norm = np.clip(lum, 0, 255) / 255.0
        grain_weight = 4.0 * lum_norm * (1.0 - lum_norm)  # peaks at midtones
        grain_weighted = grain * grain_weight
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + grain_weighted, 0, 255)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Blend with original based on strength
    if strength < 1.0:
        img = (original.astype(np.float32) * (1.0 - strength) +
               img.astype(np.float32) * strength)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def apply_basic_adjustments(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Apply Lightroom-style basic adjustments to an 8-bit BGR image."""
    img = img_bgr.astype(np.float32)

    # --- Temperature & Tint ---
    temp = params.get("temp", 0) / 100.0    # -1 warm → +1 cool
    tint = params.get("tint", 0) / 100.0    # -1 green → +1 magenta
    if temp != 0:
        img[:, :, 2] = np.clip(img[:, :, 2] + temp * 28, 0, 255)   # R (warm)
        img[:, :, 0] = np.clip(img[:, :, 0] - temp * 28, 0, 255)   # B (cool)
    if tint != 0:
        img[:, :, 1] = np.clip(img[:, :, 1] - tint * 18, 0, 255)   # G
        img[:, :, 2] = np.clip(img[:, :, 2] + tint * 9, 0, 255)    # R
        img[:, :, 0] = np.clip(img[:, :, 0] + tint * 9, 0, 255)    # B

    # --- Exposure (EV, -5 to +5) ---
    exposure = params.get("exposure", 0.0)
    if exposure != 0:
        img = img * (2.0 ** exposure)

    # --- Contrast (S-curve around midpoint) ---
    contrast = params.get("contrast", 0) / 100.0
    if contrast != 0:
        img = (img - 128.0) * (1.0 + contrast * 0.8) + 128.0

    img = np.clip(img, 0, 255)

    # --- Highlights & Shadows (luminance-masked push) ---
    highlights = params.get("highlights", 0) / 100.0
    shadows = params.get("shadows", 0) / 100.0
    whites = params.get("whites", 0) / 100.0
    blacks = params.get("blacks", 0) / 100.0

    if any(v != 0 for v in [highlights, shadows, whites, blacks]):
        lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

        if highlights != 0:
            # Smooth mask: bright midtones → highlights
            h_mask = np.clip((lum - 96.0) / 159.0, 0, 1) ** 1.5
            img = np.clip(img + h_mask[:, :, np.newaxis] * highlights * 60, 0, 255)

        if shadows != 0:
            # Smooth mask: dark midtones → shadows
            s_mask = np.clip(1.0 - lum / 160.0, 0, 1) ** 1.5
            img = np.clip(img + s_mask[:, :, np.newaxis] * shadows * 60, 0, 255)

        if whites != 0:
            # Very bright pixels only
            w_mask = np.clip((lum - 200.0) / 55.0, 0, 1)
            img = np.clip(img + w_mask[:, :, np.newaxis] * whites * 40, 0, 255)

        if blacks != 0:
            # Very dark pixels only
            b_mask = np.clip(1.0 - lum / 70.0, 0, 1)
            img = np.clip(img + b_mask[:, :, np.newaxis] * blacks * 35, 0, 255)

    # --- Vibrance (smart saturation — protects already-saturated & skin tones) ---
    vibrance = params.get("vibrance", 0) / 100.0
    if vibrance != 0:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        cur_sat = hsv[:, :, 1] / 255.0
        # Less-saturated pixels get more boost (soft-knee)
        boost_weight = (1.0 - cur_sat) * np.clip(hsv[:, :, 2] / 255.0, 0.1, 1.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + boost_weight * vibrance * 160, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # --- Saturation (global) ---
    saturation = params.get("saturation", 0) / 100.0
    if saturation != 0:
        lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        lum3 = np.stack([lum, lum, lum], axis=-1)
        img = lum3 + (img - lum3) * (1.0 + saturation)
        img = np.clip(img, 0, 255)

    # --- Clarity (midtone local contrast, ~25px radius) ---
    clarity = params.get("clarity", 0) / 100.0
    if clarity != 0:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_u8, (0, 0), 25)
        detail = img_u8.astype(np.float32) - blurred.astype(np.float32)
        lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        midtone_w = 4.0 * (lum / 255.0) * (1.0 - lum / 255.0)
        img = img + detail * clarity * 1.2 * midtone_w[:, :, np.newaxis]
        img = np.clip(img, 0, 255)

    # --- Texture (fine detail contrast, ~3px radius) ---
    texture = params.get("texture", 0) / 100.0
    if texture != 0:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_u8, (0, 0), 3)
        detail = img_u8.astype(np.float32) - blurred.astype(np.float32)
        img = img + detail * texture * 2.0
        img = np.clip(img, 0, 255)

    return np.clip(img, 0, 255).astype(np.uint8)


def build_lut_cube(source_bgr: np.ndarray, reference_bgr: np.ndarray,
                   method: str, strength: float, lut_size: int = 33,
                   film_stock: Optional[str] = None, film_strength: float = 1.0) -> str:
    """Build a .cube LUT file content string from the color transformation."""
    N = lut_size
    vals = np.linspace(0, 255, N, dtype=np.float32)

    # Build identity grid in .cube order: R fastest (axis 2), B slowest (axis 0)
    bv, gv, rv = np.meshgrid(vals, vals, vals, indexing='ij')
    r_flat = rv.ravel()  # R varies fastest after C-order ravel — matches .cube spec
    g_flat = gv.ravel()
    b_flat = bv.ravel()

    # Input LUT as BGR image (1 x N^3 x 3)
    lut_bgr = np.stack([b_flat, g_flat, r_flat], axis=-1).astype(np.uint8).reshape(1, N * N * N, 3)

    src8 = (source_bgr / 256).astype(np.uint8) if source_bgr.dtype == np.uint16 else source_bgr.copy()
    ref8 = (reference_bgr / 256).astype(np.uint8) if reference_bgr.dtype == np.uint16 else reference_bgr.copy()

    if method == "reinhard":
        src_lab = cv2.cvtColor(src8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(ref8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

        lut_f = lut_bgr.astype(np.float32) / 255.0
        lut_lab = cv2.cvtColor(lut_f, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        result_lab = lut_lab.copy()

        for ch in range(3):
            src_std = max(float(src_lab[:, :, ch].std()), 1e-6)
            src_mean = float(src_lab[:, :, ch].mean())
            ref_mean = float(ref_lab[:, :, ch].mean())
            ref_std = float(ref_lab[:, :, ch].std())
            t = (lut_lab[:, ch] - src_mean) * (ref_std / src_std) + ref_mean
            result_lab[:, ch] = lut_lab[:, ch] * (1.0 - strength) + t * strength

        result_bgr = cv2.cvtColor(
            result_lab.reshape(1, N * N * N, 3).astype(np.float32),
            cv2.COLOR_LAB2BGR
        ).reshape(-1, 3)
        result_bgr = np.clip(result_bgr * 255.0, 0, 255)

    else:  # histogram
        src_rgb = cv2.cvtColor(src8, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(ref8, cv2.COLOR_BGR2RGB)

        result_rgb = np.stack([r_flat, g_flat, b_flat], axis=-1).astype(np.float32)

        for ch in range(3):
            src_hist, _ = np.histogram(src_rgb[:, :, ch].ravel(), bins=256, range=(0, 255))
            ref_hist, _ = np.histogram(ref_rgb[:, :, ch].ravel(), bins=256, range=(0, 255))
            src_cdf = np.cumsum(src_hist).astype(np.float64)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            src_cdf /= max(src_cdf[-1], 1)
            ref_cdf /= max(ref_cdf[-1], 1)
            mapping = np.interp(src_cdf, ref_cdf, np.arange(256))
            in_vals = result_rgb[:, ch]
            out_vals = np.interp(in_vals, np.arange(256), mapping)
            if strength < 1.0:
                out_vals = in_vals * (1.0 - strength) + out_vals * strength
            result_rgb[:, ch] = out_vals

        result_rgb = np.clip(result_rgb, 0, 255)
        # Reorder to BGR for consistent handling below
        result_bgr = result_rgb[:, [2, 1, 0]]

    # Optionally apply film stock color grade (no grain/halation for LUT)
    if film_stock and film_stock in FILM_STOCKS:
        result_img = result_bgr.astype(np.uint8).reshape(1, N * N * N, 3)
        result_img = apply_film_stock(result_img, film_stock, film_strength, skip_grain=True)
        result_bgr = result_img.reshape(-1, 3).astype(np.float32)

    # Write .cube content
    title = f'LookMatch {method.capitalize()}'
    if film_stock and film_stock in FILM_STOCKS:
        title += f' + {FILM_STOCKS[film_stock]["name"]}'
    lines = [f'TITLE "{title}"', f'LUT_3D_SIZE {N}', '']
    for i in range(N * N * N):
        r_out = float(result_bgr[i, 2]) / 255.0
        g_out = float(result_bgr[i, 1]) / 255.0
        b_out = float(result_bgr[i, 0]) / 255.0
        lines.append(f'{r_out:.6f} {g_out:.6f} {b_out:.6f}')

    return '\n'.join(lines)


# --- API Routes ---

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), role: str = Form("target")):
    """Upload a photo (RAW or standard). Role is 'reference' or 'target'."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALL_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    # Stream to disk to avoid memory issues with large RAW files
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    # Generate preview
    try:
        img = read_image(str(save_path), half_size=(ext in RAW_EXTENSIONS))
        preview_img = make_preview(img)
        preview_name = save_preview(preview_img, file_id)
    except Exception as e:
        os.remove(save_path)
        if ext in RAW_EXTENSIONS:
            detail = f"Failed to process RAW file '{file.filename}': {e}"
        elif ext in HEIF_EXTENSIONS:
            detail = f"Failed to process HEIC/HEIF file '{file.filename}': {e}"
        else:
            detail = f"Failed to process '{file.filename}' ({ext}): {e}"
        raise HTTPException(500, detail)

    h, w = img.shape[:2]
    entry = {
        "id": file_id,
        "filename": file.filename,
        "path": str(save_path),
        "preview": preview_name,
        "width": w,
        "height": h,
    }

    if role == "reference":
        # Clear old reference
        if session["reference"]:
            old_path = session["reference"].get("path")
            if old_path and os.path.exists(old_path):
                os.remove(old_path)
        session["reference"] = entry
        session["matched"] = {}  # Reset matches when reference changes
    else:
        session["targets"].append(entry)

    return {
        "id": file_id,
        "filename": file.filename,
        "preview": f"/previews/{preview_name}",
        "width": w,
        "height": h,
        "role": role,
    }


@app.post("/match")
async def match_images(method: str = Form("reinhard"), strength: float = Form(85)):
    """Run color matching on all targets against the reference."""
    if not session["reference"]:
        raise HTTPException(400, "No reference image uploaded")
    if not session["targets"]:
        raise HTTPException(400, "No target images uploaded")

    strength_f = strength / 100.0
    session["settings"]["method"] = method
    session["settings"]["strength"] = strength

    # Load reference preview for matching computation
    ref_img = read_image(session["reference"]["path"], half_size=True)
    ref_preview = make_preview(ref_img)

    results = []
    for target in session["targets"]:
        try:
            tgt_img = read_image(target["path"], half_size=True)
            tgt_preview = make_preview(tgt_img)

            matched = apply_color_match(tgt_preview, ref_preview, method, strength_f)
            matched_preview_name = save_preview(matched, f"{target['id']}_matched")

            session["matched"][target["id"]] = {
                "preview": matched_preview_name,
            }

            results.append({
                "id": target["id"],
                "filename": target["filename"],
                "original_preview": f"/previews/{target['preview']}",
                "matched_preview": f"/previews/{matched_preview_name}",
            })
        except Exception as e:
            traceback.print_exc()
            results.append({
                "id": target["id"],
                "filename": target["filename"],
                "error": str(e),
            })

    return {"results": results}


@app.post("/export/{target_id}")
async def export_single(target_id: str, format: str = Form("jpeg")):
    """Export a single matched target at full resolution."""
    target = next((t for t in session["targets"] if t["id"] == target_id), None)
    if not target:
        raise HTTPException(404, "Target not found")
    if not session["reference"]:
        raise HTTPException(400, "No reference image")

    method = session["settings"]["method"]
    strength_f = session["settings"]["strength"] / 100.0

    # Full-res processing
    ref_full = read_image(session["reference"]["path"], half_size=False)
    tgt_full = read_image(target["path"], half_size=False)

    # Resize reference to match target dimensions for color stats
    ref_resized = cv2.resize(ref_full, (tgt_full.shape[1], tgt_full.shape[0]),
                             interpolation=cv2.INTER_AREA)
    matched_full = apply_color_match(tgt_full, ref_resized, method, strength_f)

    # Apply film stock if set
    film_stock = session["settings"].get("film_stock")
    film_strength_f = session["settings"].get("film_strength", 100) / 100.0
    film_overrides = session["settings"].get("film_overrides", {})
    if film_stock and film_stock in FILM_STOCKS:
        orig_params = FILM_STOCKS[film_stock].copy()
        FILM_STOCKS[film_stock].update(film_overrides)
        if matched_full.dtype == np.uint16:
            m8 = (matched_full / 256).astype(np.uint8)
            m8 = apply_film_stock(m8, film_stock, film_strength_f)
            matched_full = m8.astype(np.uint16) * 256
        else:
            matched_full = apply_film_stock(matched_full, film_stock, film_strength_f)
        FILM_STOCKS[film_stock].update(orig_params)

    # Apply basic adjustments if set
    adj = session["settings"].get("adjustments", {})
    if any(v != 0 for v in adj.values()):
        if matched_full.dtype == np.uint16:
            m8 = (matched_full / 256).astype(np.uint8)
            m8 = apply_basic_adjustments(m8, adj)
            matched_full = m8.astype(np.uint16) * 256
        else:
            matched_full = apply_basic_adjustments(matched_full, adj)

    stem = Path(target["filename"]).stem
    suffix = "_matched" if not film_stock else f"_matched_{film_stock}"

    if format == "jpeg":
        out_name = f"{stem}{suffix}.jpg"
        out_path = EXPORT_DIR / out_name
        if matched_full.dtype == np.uint16:
            matched8 = (matched_full / 256).astype(np.uint8)
        else:
            matched8 = matched_full
        cv2.imwrite(str(out_path), matched8, [cv2.IMWRITE_JPEG_QUALITY, 95])

    elif format == "tiff":
        out_name = f"{stem}{suffix}.tiff"
        out_path = EXPORT_DIR / out_name
        if matched_full.dtype != np.uint16:
            matched16 = matched_full.astype(np.uint16) * 256
        else:
            matched16 = matched_full
        rgb16 = cv2.cvtColor(matched16, cv2.COLOR_BGR2RGB)
        tifffile.imwrite(str(out_path), rgb16, photometric='rgb')

    elif format == "layered_tiff":
        out_name = f"{stem}{suffix}_layered.tiff"
        out_path = EXPORT_DIR / out_name
        if tgt_full.dtype == np.uint16:
            orig8 = cv2.cvtColor((tgt_full / 256).astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            orig8 = cv2.cvtColor(tgt_full, cv2.COLOR_BGR2RGB)
        if matched_full.dtype == np.uint16:
            match8 = cv2.cvtColor((matched_full / 256).astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            match8 = cv2.cvtColor(matched_full, cv2.COLOR_BGR2RGB)
        stacked = np.stack([orig8, match8], axis=0)
        tifffile.imwrite(str(out_path), stacked, photometric='rgb',
                         metadata={"PageName": ["Original", "Matched"]})
    else:
        raise HTTPException(400, f"Unknown format: {format}")

    return {"download": f"/exports/{out_name}", "filename": out_name}


@app.post("/export-all")
async def export_all(format: str = Form("jpeg")):
    """Export all matched targets, return zip."""
    if not session["reference"] or not session["targets"]:
        raise HTTPException(400, "Need reference and targets")

    method = session["settings"]["method"]
    strength_f = session["settings"]["strength"] / 100.0

    film_stock = session["settings"].get("film_stock")
    film_strength_f = session["settings"].get("film_strength", 100) / 100.0
    film_overrides = session["settings"].get("film_overrides", {})
    adj = session["settings"].get("adjustments", {})
    suffix = "_matched" if not film_stock else f"_matched_{film_stock}"

    ref_full = read_image(session["reference"]["path"], half_size=False)
    exported_files = []

    for target in session["targets"]:
        try:
            tgt_full = read_image(target["path"], half_size=False)
            ref_resized = cv2.resize(ref_full, (tgt_full.shape[1], tgt_full.shape[0]),
                                     interpolation=cv2.INTER_AREA)
            matched_full = apply_color_match(tgt_full, ref_resized, method, strength_f)

            if film_stock and film_stock in FILM_STOCKS:
                orig_params = FILM_STOCKS[film_stock].copy()
                FILM_STOCKS[film_stock].update(film_overrides)
                if matched_full.dtype == np.uint16:
                    m8 = (matched_full / 256).astype(np.uint8)
                    m8 = apply_film_stock(m8, film_stock, film_strength_f)
                    matched_full = m8.astype(np.uint16) * 256
                else:
                    matched_full = apply_film_stock(matched_full, film_stock, film_strength_f)
                FILM_STOCKS[film_stock].update(orig_params)

            if any(v != 0 for v in adj.values()):
                if matched_full.dtype == np.uint16:
                    m8 = (matched_full / 256).astype(np.uint8)
                    m8 = apply_basic_adjustments(m8, adj)
                    matched_full = m8.astype(np.uint16) * 256
                else:
                    matched_full = apply_basic_adjustments(matched_full, adj)

            stem = Path(target["filename"]).stem

            if format == "jpeg":
                out_name = f"{stem}{suffix}.jpg"
                out_path = EXPORT_DIR / out_name
                if matched_full.dtype == np.uint16:
                    m8 = (matched_full / 256).astype(np.uint8)
                else:
                    m8 = matched_full
                cv2.imwrite(str(out_path), m8, [cv2.IMWRITE_JPEG_QUALITY, 95])

            elif format == "tiff":
                out_name = f"{stem}{suffix}.tiff"
                out_path = EXPORT_DIR / out_name
                if matched_full.dtype != np.uint16:
                    m16 = matched_full.astype(np.uint16) * 256
                else:
                    m16 = matched_full
                rgb16 = cv2.cvtColor(m16, cv2.COLOR_BGR2RGB)
                tifffile.imwrite(str(out_path), rgb16, photometric='rgb')

            elif format == "layered_tiff":
                out_name = f"{stem}{suffix}_layered.tiff"
                out_path = EXPORT_DIR / out_name
                if tgt_full.dtype == np.uint16:
                    orig8 = cv2.cvtColor((tgt_full / 256).astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    orig8 = cv2.cvtColor(tgt_full, cv2.COLOR_BGR2RGB)
                if matched_full.dtype == np.uint16:
                    match8 = cv2.cvtColor((matched_full / 256).astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    match8 = cv2.cvtColor(matched_full, cv2.COLOR_BGR2RGB)
                stacked = np.stack([orig8, match8], axis=0)
                tifffile.imwrite(str(out_path), stacked, photometric='rgb',
                                 metadata={"PageName": ["Original", "Matched"]})

            exported_files.append(out_name)
            # Free memory
            del tgt_full, matched_full
        except Exception as e:
            traceback.print_exc()
            exported_files.append(f"FAILED: {target['filename']} - {str(e)}")

    # Free reference from memory
    del ref_full

    # Create zip
    zip_name = "lookmatch_export.zip"
    zip_path = EXPORT_DIR / zip_name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in exported_files:
            if not fname.startswith("FAILED"):
                fpath = EXPORT_DIR / fname
                if fpath.exists():
                    zf.write(fpath, fname)

    return {"download": f"/exports/{zip_name}", "files": exported_files}


@app.post("/save-profile")
async def save_profile():
    """Save the reference image's LAB statistics as a reusable profile."""
    if not session["reference"]:
        raise HTTPException(400, "No reference image")

    ref_img = read_image(session["reference"]["path"], half_size=True)
    ref_preview = make_preview(ref_img)
    ref_lab = cv2.cvtColor(ref_preview.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

    profile = {
        "name": Path(session["reference"]["filename"]).stem,
        "channels": {}
    }
    for i, ch_name in enumerate(["L", "A", "B"]):
        ch = ref_lab[:, :, i]
        profile["channels"][ch_name] = {
            "mean": float(ch.mean()),
            "std": float(ch.std()),
        }

    profile_name = f"{profile['name']}_profile.json"
    profile_path = EXPORT_DIR / profile_name
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    return {"download": f"/exports/{profile_name}", "profile": profile}


@app.post("/clear")
async def clear_session():
    """Clear all uploads and reset session."""
    for d in [UPLOAD_DIR, PREVIEW_DIR, EXPORT_DIR]:
        for f in d.iterdir():
            f.unlink()
    session["reference"] = None
    session["targets"] = []
    session["matched"] = {}
    session["settings"]["film_stock"] = None
    session["settings"]["film_strength"] = 100
    session["settings"]["film_overrides"] = {}
    session["settings"]["adjustments"] = {
        "temp": 0, "tint": 0, "exposure": 0.0, "contrast": 0,
        "highlights": 0, "shadows": 0, "whites": 0, "blacks": 0,
        "vibrance": 0, "saturation": 0, "clarity": 0, "texture": 0,
    }
    return {"status": "cleared"}


@app.delete("/target/{target_id}")
async def remove_target(target_id: str):
    """Remove a single target."""
    target = next((t for t in session["targets"] if t["id"] == target_id), None)
    if target:
        if os.path.exists(target["path"]):
            os.remove(target["path"])
        session["targets"] = [t for t in session["targets"] if t["id"] != target_id]
        session["matched"].pop(target_id, None)
    return {"status": "removed"}


@app.get("/film-stocks")
async def get_film_stocks():
    """Return available film stock presets including their default parameter values."""
    return {
        k: {
            "name": v["name"],
            "description": v["description"],
            "bw": v.get("bw", False),
            "defaults": {
                "grain_amount":   v.get("grain_amount", 0),
                "grain_size":     v.get("grain_size", 1.0),
                "grain_roughness": v.get("grain_roughness", 0.0),
                "fade":           v.get("fade", 0),
                "contrast":       v.get("contrast", 1.0),
                "saturation":     v.get("saturation", 1.0),
                "halation":       v.get("halation", 0),
            },
        }
        for k, v in FILM_STOCKS.items()
    }


@app.post("/apply-film")
async def apply_film(
    stock: str = Form(...),
    strength: float = Form(100),
    grain_amount: Optional[float] = Form(None),
    grain_size: Optional[float] = Form(None),
    grain_roughness: Optional[float] = Form(None),
    fade: Optional[float] = Form(None),
    halation: Optional[float] = Form(None),
    contrast: Optional[float] = Form(None),
    saturation_film: Optional[float] = Form(None),
):
    """Apply film stock emulation preview to all targets."""
    if not session["targets"]:
        raise HTTPException(400, "No target images")
    if stock not in FILM_STOCKS:
        raise HTTPException(400, f"Unknown film stock: {stock}")

    strength_f = strength / 100.0
    session["settings"]["film_stock"] = stock
    session["settings"]["film_strength"] = strength

    # Store any overrides
    overrides = {}
    if grain_amount is not None:
        overrides["grain_amount"] = grain_amount
    if grain_size is not None:
        overrides["grain_size"] = grain_size
    if grain_roughness is not None:
        overrides["grain_roughness"] = grain_roughness
    if fade is not None:
        overrides["fade"] = fade
    if halation is not None:
        overrides["halation"] = halation
    if contrast is not None:
        overrides["contrast"] = contrast
    if saturation_film is not None:
        overrides["saturation"] = saturation_film
    session["settings"]["film_overrides"] = overrides

    # Build effective params (stock defaults + overrides)
    effective = {**FILM_STOCKS[stock], **overrides}

    results = []
    for target in session["targets"]:
        try:
            matched_info = session["matched"].get(target["id"], {})
            base_preview = matched_info.get("preview") or target["preview"]
            base_path = PREVIEW_DIR / base_preview
            base_img = cv2.imread(str(base_path))
            if base_img is None:
                raise ValueError("Could not load preview")

            # Apply with effective params (temporarily patch stock)
            orig = FILM_STOCKS[stock].copy()
            FILM_STOCKS[stock].update(overrides)
            film_img = apply_film_stock(base_img, stock, strength_f)
            FILM_STOCKS[stock].update(orig)

            film_preview_name = save_preview(film_img, f"{target['id']}_film")
            if target["id"] not in session["matched"]:
                session["matched"][target["id"]] = {}
            session["matched"][target["id"]]["film_preview"] = film_preview_name

            results.append({
                "id": target["id"],
                "film_preview": f"/previews/{film_preview_name}",
            })
        except Exception as e:
            traceback.print_exc()
            results.append({"id": target["id"], "error": str(e)})

    return {
        "results": results,
        "stock": stock,
        "stock_name": FILM_STOCKS[stock]["name"],
        "defaults": {k: FILM_STOCKS[stock].get(k) for k in
                     ["grain_amount", "grain_size", "fade", "halation", "contrast", "saturation"]},
    }


@app.post("/apply-adjustments")
async def apply_adjustments_endpoint(
    temp: float = Form(0), tint: float = Form(0),
    exposure: float = Form(0.0), contrast: float = Form(0),
    highlights: float = Form(0), shadows: float = Form(0),
    whites: float = Form(0), blacks: float = Form(0),
    vibrance: float = Form(0), saturation: float = Form(0),
    clarity: float = Form(0), texture: float = Form(0),
):
    """Apply Lightroom-style basic adjustments to all target previews."""
    if not session["targets"]:
        raise HTTPException(400, "No target images")

    adj = {
        "temp": temp, "tint": tint,
        "exposure": exposure, "contrast": contrast,
        "highlights": highlights, "shadows": shadows,
        "whites": whites, "blacks": blacks,
        "vibrance": vibrance, "saturation": saturation,
        "clarity": clarity, "texture": texture,
    }
    session["settings"]["adjustments"] = adj

    results = []
    for target in session["targets"]:
        try:
            matched_info = session["matched"].get(target["id"], {})
            # Apply on top of film preview if available, else matched, else original
            base_preview = (matched_info.get("film_preview") or
                            matched_info.get("preview") or
                            target["preview"])
            base_img = cv2.imread(str(PREVIEW_DIR / base_preview))
            if base_img is None:
                raise ValueError("Could not load preview")

            adjusted = apply_basic_adjustments(base_img, adj)
            adj_name = save_preview(adjusted, f"{target['id']}_adj")
            session["matched"].setdefault(target["id"], {})
            session["matched"][target["id"]]["adj_preview"] = adj_name

            results.append({"id": target["id"], "adj_preview": f"/previews/{adj_name}"})
        except Exception as e:
            traceback.print_exc()
            results.append({"id": target["id"], "error": str(e)})

    return {"results": results}


@app.post("/clear-film")
async def clear_film():
    """Remove film stock from session."""
    session["settings"]["film_stock"] = None
    session["settings"]["film_strength"] = 100
    # Remove film previews from matched dict
    for tid in session["matched"]:
        session["matched"][tid].pop("film_preview", None)
    return {"status": "cleared"}


@app.post("/export-lut")
async def export_lut(lut_size: int = Form(33)):
    """Export a .cube LUT capturing the current color match transformation."""
    if not session["reference"]:
        raise HTTPException(400, "No reference image")
    if not session["targets"]:
        raise HTTPException(400, "No target images")

    method = session["settings"]["method"]
    strength_f = session["settings"]["strength"] / 100.0
    film_stock = session["settings"].get("film_stock")
    film_strength_f = session["settings"].get("film_strength", 100) / 100.0

    target = session["targets"][0]
    ref_img = read_image(session["reference"]["path"], half_size=True)
    ref8 = make_preview(ref_img)
    tgt_img = read_image(target["path"], half_size=True)
    tgt8 = make_preview(tgt_img)
    ref8_resized = cv2.resize(ref8, (tgt8.shape[1], tgt8.shape[0]), interpolation=cv2.INTER_AREA)

    cube_content = build_lut_cube(tgt8, ref8_resized, method, strength_f,
                                  lut_size, film_stock, film_strength_f)

    lut_name = f"lookmatch_grade.cube"
    lut_path = EXPORT_DIR / lut_name
    with open(lut_path, "w") as f:
        f.write(cube_content)

    return {"download": f"/exports/{lut_name}", "filename": lut_name}


@app.post("/export-film-lut")
async def export_film_lut(stock: str = Form(...), strength: float = Form(100),
                          lut_size: int = Form(33)):
    """Export a .cube LUT for a film stock color grade (no grain/halation)."""
    if stock not in FILM_STOCKS:
        raise HTTPException(400, f"Unknown film stock: {stock}")

    strength_f = strength / 100.0
    N = lut_size
    vals = np.linspace(0, 255, N, dtype=np.float32)

    # Identity LUT in .cube order
    bv, gv, rv = np.meshgrid(vals, vals, vals, indexing='ij')
    r_flat = rv.ravel()
    g_flat = gv.ravel()
    b_flat = bv.ravel()
    lut_bgr = np.stack([b_flat, g_flat, r_flat], axis=-1).astype(np.uint8).reshape(1, N * N * N, 3)

    result = apply_film_stock(lut_bgr, stock, strength_f, skip_grain=True)
    result_flat = result.reshape(-1, 3).astype(np.float32)

    stock_name = FILM_STOCKS[stock]["name"]
    lines = [f'TITLE "{stock_name} (LookMatch)"', f'LUT_3D_SIZE {N}', '']
    for i in range(N * N * N):
        r_out = float(result_flat[i, 2]) / 255.0
        g_out = float(result_flat[i, 1]) / 255.0
        b_out = float(result_flat[i, 0]) / 255.0
        lines.append(f'{r_out:.6f} {g_out:.6f} {b_out:.6f}')

    cube_content = '\n'.join(lines)
    lut_name = f"lookmatch_{stock}.cube"
    lut_path = EXPORT_DIR / lut_name
    with open(lut_path, "w") as f:
        f.write(cube_content)

    return {"download": f"/exports/{lut_name}", "filename": lut_name}


# Serve static files
app.mount("/previews", StaticFiles(directory=str(PREVIEW_DIR)), name="previews")
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 50)
    print(f"  LookMatch running at http://localhost:{port}")
    if _PASSWORD:
        print("  Password protection: ON")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
