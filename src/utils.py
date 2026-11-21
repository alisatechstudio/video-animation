"""
Utilities for FFmpeg discovery, session directories, model caching, and cleanup.
"""
import os
import shutil
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from src.config import TEMP_DIR, MODELS_DIR

# Known reliable ONNX model download URLs (AnimeGANv2)
MODEL_URLS = {
    "animegan2_hayao.onnx": "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/face_paint_512_v2.pt", # PyTorch or ONNX
    # We can also download the direct onnx model from Hugging Face / GitHub
    "AnimeGANv2_Hayao.onnx": "https://huggingface.co/sayakpaul/AnimeGANv2-ONNX/resolve/main/AnimeGANv2_Hayao.onnx",
    "AnimeGANv2_Shinkai.onnx": "https://huggingface.co/sayakpaul/AnimeGANv2-ONNX/resolve/main/AnimeGANv2_Shinkai.onnx",
}


def get_ffmpeg_path() -> str:
    """
    Returns the absolute path to a working ffmpeg binary.
    Prioritizes system PATH, falls back to imageio-ffmpeg.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg and os.path.isfile(system_ffmpeg):
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
            return ffmpeg_exe
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg executable not found. Please install ffmpeg or ensure 'imageio-ffmpeg' is installed."
    )


def create_session_dirs(session_id: Optional[str] = None) -> Dict[str, Path]:
    """
    Creates and returns isolated working directories for a processing session.
    """
    if not session_id:
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    session_root = TEMP_DIR / session_id
    raw_frames = session_root / "raw_frames"
    stylized_frames = session_root / "stylized_frames"
    audio_dir = session_root / "audio"

    for d in [raw_frames, stylized_frames, audio_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "id": session_id,
        "root": session_root,
        "raw_frames": raw_frames,
        "stylized_frames": stylized_frames,
        "audio": audio_dir
    }


def cleanup_session(session_root: Path) -> bool:
    """
    Removes session directory with Windows-safe retry logic.
    """
    if not session_root.exists():
        return True

    for _ in range(3):
        try:
            shutil.rmtree(session_root, ignore_errors=False)
            return True
        except Exception:
            time.sleep(0.3)

    try:
        shutil.rmtree(session_root, ignore_errors=True)
        return not session_root.exists()
    except Exception:
        return False


def cleanup_all_temp():
    """
    Cleans up all session directories inside the temp folder.
    """
    if not TEMP_DIR.exists():
        return

    for item in TEMP_DIR.iterdir():
        if item.is_dir() and item.name.startswith("sess_"):
            cleanup_session(item)


def ensure_onnx_model(model_filename: str) -> Optional[Path]:
    """
    Checks if an ONNX model exists in MODELS_DIR; downloads it from Hugging Face if missing.
    Returns None if download fails (will fallback gracefully to artistic CV shaders).
    """
    target_path = MODELS_DIR / model_filename
    if target_path.exists() and target_path.stat().st_size > 100_000:
        return target_path

    url = MODEL_URLS.get(model_filename)
    if not url:
        return None

    try:
        print(f"Downloading model {model_filename} from {url}...")
        urllib.request.urlretrieve(url, target_path)
        if target_path.exists() and target_path.stat().st_size > 100_000:
            print(f"Successfully cached {model_filename} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
            return target_path
    except Exception as e:
        print(f"Could not download {model_filename}: {e}. Fallback shader will be used.")
        if target_path.exists():
            target_path.unlink()

    return None
