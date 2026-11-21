"""
Configuration, Hardware Detection, and Preset Definitions
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# Root Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class HardwareSpecs:
    has_cuda: bool
    device_name: str
    vram_gb: float
    recommended_mode: str  # "diffusion" or "lightweight"
    warning_message: str = ""


def detect_hardware() -> HardwareSpecs:
    """
    Detects GPU, CUDA support, and available VRAM.
    Adapts recommendation based on whether VRAM >= 8GB or CPU-only.
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 2)
            device_name = torch.cuda.get_device_name(0)
            if vram_gb >= 8.0:
                return HardwareSpecs(
                    has_cuda=True,
                    device_name=device_name,
                    vram_gb=vram_gb,
                    recommended_mode="diffusion",
                    warning_message=f"Optimal GPU detected: {device_name} with {vram_gb} GB VRAM."
                )
            else:
                return HardwareSpecs(
                    has_cuda=True,
                    device_name=device_name,
                    vram_gb=vram_gb,
                    recommended_mode="lightweight",
                    warning_message=(
                        f"GPU has {vram_gb} GB VRAM (< 8GB threshold). "
                        "Lightweight AnimeGAN/Artistic mode is recommended for fast processing."
                    )
                )
    except Exception:
        pass

    # CPU Fallback
    return HardwareSpecs(
        has_cuda=False,
        device_name="CPU / Integrated Graphics",
        vram_gb=0.0,
        recommended_mode="lightweight",
        warning_message="CPU Mode Active: High-speed AnimeGANv2 & Artistic Shader Engine enabled for fluid rendering."
    )


# Style Presets
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Studio Ghibli Anime": {
        "engine": "animegan2",
        "model_variant": "hayao",
        "positive_prompt": "masterpiece, Studio Ghibli style anime aesthetic, lush hand-painted background, soft natural lighting, Hayao Miyazaki aesthetic, cel shading, highly detailed",
        "negative_prompt": "photorealistic, hyperrealistic, 3d render, noisy, low quality, oversaturated, blurry, bad anatomy",
        "description": "Lush, pastoral anime aesthetics reminiscent of classic Studio Ghibli films."
    },
    "Makoto Shinkai Anime": {
        "engine": "animegan2",
        "model_variant": "shinkai",
        "positive_prompt": "masterpiece, Makoto Shinkai style anime, vibrant cinematic sky, volumetric cloud lighting, hyper-detailed anime aesthetic, vivid colors, emotional lighting",
        "negative_prompt": "dark, dull, muddy, low quality, noisy, ugly, photorealistic",
        "description": "Vibrant, high-contrast cinematic anime with dramatic lighting and sky tones."
    },
    "Cyberpunk Neon": {
        "engine": "artistic",
        "shader": "cyberpunk",
        "positive_prompt": "cyberpunk 2077 aesthetic, glowing neon cyan and magenta outlines, synthwave lighting, futuristic dark sci-fi city style, high contrast, sharp edge glow",
        "negative_prompt": "sepia, vintage, muted colors, low resolution, daylight, washed out",
        "description": "High-contrast neon contour glow, deep cyan/magenta grade, and synthwave atmosphere."
    },
    "Watercolor Painting": {
        "engine": "artistic",
        "shader": "watercolor",
        "positive_prompt": "exquisite watercolor painting, flowing soft pigment edges, paper texture grain, wet-on-wet paint bleeding, delicate color washes, artistic masterpiece",
        "negative_prompt": "digital harsh edges, glossy, 3d render, plastic, oversaturated",
        "description": "Soft multi-layer pigment diffusion, paper grain texture, and organic brush blending."
    },
    "Comic Book Ink": {
        "engine": "artistic",
        "shader": "comic",
        "positive_prompt": "vintage comic book style, bold black ink outlines, halftone dot shading, retro graphic novel, cel-shaded color blocks, pop art print",
        "negative_prompt": "smooth gradient, blurry, photograph, realistic, low contrast",
        "description": "Bold ink contours, halftone Ben-Day dot shading, and punchy pop-art colors."
    },
    "Custom SD ControlNet Canny": {
        "engine": "diffusion",
        "model_id": "runwayml/stable-diffusion-v1-5",
        "controlnet_id": "lllyasviel/sd-controlnet-canny",
        "positive_prompt": "anime keyframe illustration, detailed lineart, clean digital painting, 4k masterpiece",
        "negative_prompt": "ugly, deformed, bad eyes, disfigured, blurry, low quality, duplicate",
        "description": "Deep generative diffusion via ControlNet Canny. (Requires >=8GB VRAM GPU for real-time speed)."
    }
}
