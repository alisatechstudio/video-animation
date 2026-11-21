"""
YouTube Video and Audio Downloader with Trimming Support via yt-dlp & FFmpeg.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import yt_dlp

from src.utils import get_ffmpeg_path


def download_youtube_clip(
    url: str,
    start_time: float,
    duration: float,
    output_dir: Path,
    max_height: int = 720,
    progress_callback: Optional[Any] = None
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    """
    Downloads and precisely trims a YouTube video clip and extracts its audio.

    Args:
        url: The YouTube video URL.
        start_time: Start time in seconds.
        duration: Duration of clip in seconds.
        output_dir: Directory where the media files will be saved.
        max_height: Maximum vertical resolution (e.g. 720).
        progress_callback: Optional callable for UI progress updates.

    Returns:
        Tuple of (trimmed_video_path, audio_path, metadata_dict)
    """
    if progress_callback:
        progress_callback(0.1, "Fetching video metadata...")

    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = get_ffmpeg_path()
    ffmpeg_dir = str(Path(ffmpeg_path).parent)

    end_time = start_time + duration
    raw_video_template = str(output_dir / "raw_download.%(ext)s")

    # yt-dlp options
    ydl_opts = {
        'ffmpeg_location': ffmpeg_dir,
        'format': f'bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]/best',
        'outtmpl': raw_video_template,
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
        'merge_output_format': 'mp4',
    }

    # Use yt-dlp download sections when available to save bandwidth
    try:
        ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(start_time, end_time)])
        ydl_opts['force_keyframes_at_cuts'] = True
    except Exception:
        pass

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if progress_callback:
            progress_callback(0.2, "Downloading video segment from YouTube...")
        info = ydl.extract_info(url, download=True)
        video_title = info.get('title', 'YouTube Video')
        fps = info.get('fps', 30.0)

    # Locate the downloaded file
    downloaded_files = list(output_dir.glob("raw_download.*"))
    if not downloaded_files:
        raise RuntimeError("yt-dlp completed but no downloaded file was found.")
    raw_video_path = downloaded_files[0]

    # Perform exact FFmpeg trimming for 100% millisecond precision & audio sync
    if progress_callback:
        progress_callback(0.35, "Trimming exact clip duration with FFmpeg...")

    final_trimmed_video = output_dir / "trimmed_input.mp4"
    audio_path = output_dir / "original_audio.aac"

    # Precise cut with re-encoding to guarantee clean keyframe at t=0
    trim_cmd = [
        ffmpeg_path,
        "-y",
        "-ss", str(start_time),
        "-i", str(raw_video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(final_trimmed_video)
    ]
    subprocess.run(trim_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Extract audio track
    extract_audio_cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(final_trimmed_video),
        "-vn",
        "-c:a", "aac",
        "-b:a", "192k",
        str(audio_path)
    ]
    subprocess.run(extract_audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        extracted_audio = None
    else:
        extracted_audio = audio_path

    # Clean up raw untrimmed file
    if raw_video_path.exists() and raw_video_path != final_trimmed_video:
        try:
            raw_video_path.unlink()
        except Exception:
            pass

    metadata = {
        "title": video_title,
        "fps": fps,
        "start_time": start_time,
        "duration": duration,
        "has_audio": extracted_audio is not None
    }

    return final_trimmed_video, extracted_audio, metadata
