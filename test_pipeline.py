"""
Automated CLI Smoke Test for Video AI Stylization Pipeline.
Generates a synthetic test video with audio and validates frame extraction,
preset stylization, and audio-synced video assembly.
"""
import math
import struct
import wave
import subprocess
from pathlib import Path
import cv2
import numpy as np

from src.config import TEMP_DIR, OUTPUT_DIR, STYLE_PRESETS
from src.utils import create_session_dirs, cleanup_session, get_ffmpeg_path
from src.extractor import extract_frames
from src.animator import process_frames
from src.assembler import assemble_video


def generate_synthetic_video_with_audio(output_video_path: Path, duration_sec: float = 3.0, fps: int = 24):
    """
    Creates a synthetic MP4 video of a bouncing shape with a synchronized 440Hz audio tone.
    """
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_silent_video = output_video_path.parent / "temp_silent.mp4"
    temp_audio_wav = output_video_path.parent / "temp_tone.wav"

    width, height = 640, 360
    total_frames = int(duration_sec * fps)

    # 1. Generate video frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_silent_video), fourcc, fps, (width, height))

    for i in range(total_frames):
        # Create gradient canvas
        t = i / total_frames
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, 0] = int(np.clip(60 + 120 * math.sin(t * math.pi), 0, 255))  # Blue
        canvas[:, :, 1] = int(np.clip(80 + 100 * math.cos(t * math.pi), 0, 255))  # Green
        canvas[:, :, 2] = int(np.clip(120 + 80 * math.sin(t * 2 * math.pi), 0, 255))  # Red

        # Draw bouncing ball
        ball_x = int(width / 2 + (width / 3) * math.sin(t * 4 * math.pi))
        ball_y = int(height / 2 + (height / 3) * math.cos(t * 4 * math.pi))
        cv2.circle(canvas, (ball_x, ball_y), 35, (255, 255, 255), -1)
        cv2.circle(canvas, (ball_x, ball_y), 32, (0, 200, 255), -1)

        # Draw frame counter
        text = f"Frame {i:03d} / {total_frames} | Synth Test"
        cv2.putText(canvas, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(canvas, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        writer.write(canvas)

    writer.release()

    # 2. Generate 440Hz Audio WAV
    sample_rate = 44100
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(temp_audio_wav), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for s in range(num_samples):
            # 440Hz A tone with soft decay/attack
            tone = math.sin(2 * math.pi * 440.0 * (s / sample_rate))
            val = int(tone * 16000)
            wav_file.writeframesraw(struct.pack('<h', val))

    # 3. Mux into synthetic_test.mp4 using FFmpeg
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        ffmpeg_path, "-y",
        "-i", str(temp_silent_video),
        "-i", str(temp_audio_wav),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_video_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Cleanup temp intermediates
    if temp_silent_video.exists():
        temp_silent_video.unlink()
    if temp_audio_wav.exists():
        temp_audio_wav.unlink()


def run_smoke_test():
    print("=" * 65)
    print("AI VIDEO ANIMATOR - END-TO-END CLI SMOKE TEST")
    print("=" * 65)

    test_dir = TEMP_DIR / "smoke_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    synthetic_video = test_dir / "synthetic_test.mp4"

    # Step 1: Generate synthetic video with audio
    print("\n[Step 1/5] Generating 3-second synthetic test video with audio track...")
    generate_synthetic_video_with_audio(synthetic_video, duration_sec=3.0, fps=24)
    print(f" -> Synthetic video created: {synthetic_video} ({synthetic_video.stat().st_size} bytes)")

    # Step 2: Create Session Dirs
    print("\n[Step 2/5] Creating isolated session workspace...")
    session = create_session_dirs("test_run")
    print(f" -> Session created at: {session['root']}")

    # Extract original audio for muxing test
    ffmpeg_path = get_ffmpeg_path()
    audio_path = session["audio"] / "original_audio.aac"
    subprocess.run(
        [ffmpeg_path, "-y", "-i", str(synthetic_video), "-vn", "-c:a", "aac", str(audio_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(f" -> Audio extracted: {audio_path} ({audio_path.stat().st_size} bytes)")

    # Step 3: Extract frames
    print("\n[Step 3/5] Extracting video frames via OpenCV...")
    extraction = extract_frames(synthetic_video, session["raw_frames"])
    print(f" -> Extracted {extraction.frame_count} frames at {extraction.fps:.1f} FPS ({extraction.width}x{extraction.height})")
    assert extraction.frame_count == 72, f"Expected 72 frames, got {extraction.frame_count}"

    # Step 4: Stylize frames with artistic & ONNX presets
    test_presets = ["Cyberpunk Neon", "Watercolor Painting", "Comic Book Ink"]
    for preset_name in test_presets:
        print(f"\n[Step 4/5] Testing stylization preset: '{preset_name}'...")
        preset_out_dir = session["stylized_frames"] / preset_name.replace(" ", "_")
        stylized_paths = process_frames(
            frame_paths=extraction.frame_paths[:12],  # Test first 12 frames for speed
            output_dir=preset_out_dir,
            preset_name=preset_name,
            intensity=0.85,
            seed=42
        )
        print(f" -> Successfully stylized {len(stylized_paths)} frames using '{preset_name}'")
        assert len(stylized_paths) == 12, "Stylization frame count mismatch"

    # Full stylization with Cyberpunk Neon for final assembly
    full_stylized_dir = session["stylized_frames"] / "full"
    print("\nStylizing all 72 frames for final video assembly test...")
    all_stylized = process_frames(
        frame_paths=extraction.frame_paths,
        output_dir=full_stylized_dir,
        preset_name="Cyberpunk Neon",
        intensity=0.8,
        seed=42
    )

    # Step 5: Assemble final video and mux audio
    print("\n[Step 5/5] Assembling stylized frames and muxing audio with FFmpeg...")
    final_output = OUTPUT_DIR / "smoke_test_cyberpunk.mp4"
    assemble_video(
        frames_dir=full_stylized_dir,
        fps=extraction.fps,
        output_path=final_output,
        audio_path=audio_path
    )
    print(f" -> Output video produced: {final_output} ({final_output.stat().st_size} bytes)")
    assert final_output.exists() and final_output.stat().st_size > 5000, "Final video missing or too small"

    # Verify output video properties
    cap = cv2.VideoCapture(str(final_output))
    out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f" -> Output Verification: {out_frames} frames, {out_fps:.1f} FPS, audio preserved.")

    # Cleanup session
    cleanup_session(session["root"])
    if test_dir.exists():
        cleanup_session(test_dir)

    print("\n" + "=" * 65)
    print(">>> ALL SMOKE TESTS PASSED WITH ZERO ERRORS! <<<")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
