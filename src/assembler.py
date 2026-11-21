"""
Video Reassembly and Audio Muxing using FFmpeg.
"""
import subprocess
from pathlib import Path
from typing import Optional, Any
from src.utils import get_ffmpeg_path


def assemble_video(
    frames_dir: Path,
    fps: float,
    output_path: Path,
    audio_path: Optional[Path] = None,
    progress_callback: Optional[Any] = None
) -> Path:
    """
    Stitches stylized image frames into an H.264 MP4 video and muxes the original audio.

    Args:
        frames_dir: Directory containing stylized_%05d.png frames.
        fps: Playback frames per second.
        output_path: Destination path for the final MP4 video.
        audio_path: Optional path to the original extracted audio track.
        progress_callback: Optional progress reporter callable.

    Returns:
        Path to the assembled MP4 video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = get_ffmpeg_path()

    input_frame_pattern = str(frames_dir / "stylized_%05d.png")

    if progress_callback:
        progress_callback(0.92, "Muxing video frames and original audio...")

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate", str(fps),
        "-i", input_frame_pattern,
    ]

    # Add audio input if present
    if audio_path and Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
        cmd.extend([
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "19",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path)
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "19",
            "-preset", "fast",
            str(output_path)
        ])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg assembly failed: {result.stderr}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output video was not generated properly at {output_path}")

    if progress_callback:
        progress_callback(1.0, "Video rendering complete!")

    return output_path
