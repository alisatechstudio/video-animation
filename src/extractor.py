"""
Video Frame Extractor using OpenCV.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any
import cv2


@dataclass
class ExtractionResult:
    frame_paths: List[Path]
    fps: float
    frame_count: int
    width: int
    height: int


def extract_frames(
    video_path: Path,
    output_dir: Path,
    max_dim: Optional[int] = None,
    progress_callback: Optional[Any] = None
) -> ExtractionResult:
    """
    Extracts all frames from a video file into an output directory as PNG images.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where frames will be saved.
        max_dim: Optional maximum width or height to rescale to (ensuring even dimensions).
        progress_callback: Optional callback for UI progress reporting.

    Returns:
        ExtractionResult containing frame paths, fps, frame count, width, and height.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths: List[Path] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional aspect-ratio-preserving downscale for faster CPU inference
        if max_dim and (orig_width > max_dim or orig_height > max_dim):
            scale = max_dim / max(orig_width, orig_height)
            new_w = int(orig_width * scale)
            new_h = int(orig_height * scale)
            # Ensure even dimensions for H.264 encoder compatibility
            new_w = new_w if new_w % 2 == 0 else new_w - 1
            new_h = new_h if new_h % 2 == 0 else new_h - 1
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_filename = output_dir / f"frame_{idx:05d}.png"
        cv2.imwrite(str(frame_filename), frame)
        frame_paths.append(frame_filename)

        idx += 1
        if progress_callback and total_frames > 0 and idx % 5 == 0:
            pct = min(1.0, idx / total_frames)
            progress_callback(pct, f"Extracting frame {idx}/{total_frames}...")

    cap.release()

    if not frame_paths:
        raise RuntimeError("No frames could be extracted from video.")

    # Read final dimensions from the first saved frame
    sample_frame = cv2.imread(str(frame_paths[0]))
    h, w = sample_frame.shape[:2]

    return ExtractionResult(
        frame_paths=frame_paths,
        fps=fps,
        frame_count=len(frame_paths),
        width=w,
        height=h
    )
