"""
YouTube Video and Audio Downloader with Trimming Support via yt-dlp & FFmpeg.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import yt_dlp

from src.utils import get_ffmpeg_path
from src.config import BASE_DIR


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

    # Check for cookies (local file or environment variable)
    cookie_file = None
    local_cookie = BASE_DIR / "cookies.txt"
    if local_cookie.exists() and local_cookie.stat().st_size > 0:
        cookie_file = str(local_cookie)
    elif os.environ.get("YOUTUBE_COOKIES"):
        env_cookie = output_dir / "env_cookies.txt"
        env_cookie.write_text(os.environ["YOUTUBE_COOKIES"], encoding="utf-8")
        cookie_file = str(env_cookie)

    # Base yt-dlp options with multi-client emulation to bypass cloud bot filters
    ydl_opts = {
        'ffmpeg_location': ffmpeg_dir,
        'format': f'bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]/best',
        'outtmpl': raw_video_template,
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
        'merge_output_format': 'mp4',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb', 'web'],
                'player_skip': ['configs', 'webpage']
            }
        }
    }

    if cookie_file:
        ydl_opts['cookiefile'] = cookie_file

    # Attempt download with section trimming first
    try:
        ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(start_time, end_time)])
        ydl_opts['force_keyframes_at_cuts'] = True
    except Exception:
        pass

    try:
        if progress_callback:
            progress_callback(0.2, "Connecting to YouTube stream...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'YouTube Video')
            fps = info.get('fps', 30.0)
    except Exception as e:
        err_msg = str(e)
        if "Sign in to confirm you’re not a bot" in err_msg or "bot" in err_msg.lower():
            # Retry with pure android client fallback
            try:
                print("Retrying download with dedicated mobile client...")
                ydl_opts['extractor_args'] = {'youtube': {'player_client': ['android']}}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_title = info.get('title', 'YouTube Video')
                    fps = info.get('fps', 30.0)
            except Exception:
                raise RuntimeError(
                    "YouTube anti-bot verification active on this cloud server IP ('Sign in to confirm you’re not a bot').\n\n"
                    "💡 Quick Solution: Download the clip or use any video on your device, then switch to the 'Upload Local Video' tab to process it instantly without bot verification!"
                )
        else:
            raise RuntimeError(f"YouTube download failed: {err_msg}")

    # Locate the downloaded file
    downloaded_files = list(output_dir.glob("raw_download.*"))
    if not downloaded_files:
        raise RuntimeError("yt-dlp completed but no downloaded file was found on disk.")
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
