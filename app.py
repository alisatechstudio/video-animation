"""
YouTube to AI Animation Web Application.
Built with Gradio, OpenCV, AnimeGANv2 / Artistic Shaders / SD ControlNet, and FFmpeg.
"""
import time
import os
from pathlib import Path
from typing import Optional, Tuple
import gradio as gr

from src.config import (
    STYLE_PRESETS,
    detect_hardware,
    OUTPUT_DIR,
    TEMP_DIR,
)
from src.utils import (
    create_session_dirs,
    cleanup_session,
    cleanup_all_temp,
    get_ffmpeg_path,
)
from src.downloader import download_youtube_clip
from src.extractor import extract_frames
from src.animator import process_frames
from src.assembler import assemble_video

# Pre-detect system hardware on startup
HARDWARE = detect_hardware()


def run_pipeline(
    youtube_url: Optional[str],
    uploaded_video: Optional[str],
    start_time: float,
    duration: float,
    preset_name: str,
    intensity: float,
    custom_prompt: Optional[str],
    max_resolution: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """
    Main processing workflow orchestrating:
    1. Input acquisition (YouTube download or local upload trim)
    2. Frame & audio extraction
    3. AI stylization with progress callbacks
    4. Video assembly with original audio preservation
    5. Automatic session cleanup
    """
    if not youtube_url and not uploaded_video:
        raise gr.Error("Please provide either a YouTube video URL or upload a local video file.")

    session = create_session_dirs()
    session_root = session["root"]

    try:
        max_dim = int(max_resolution.replace("p", ""))
    except Exception:
        max_dim = 720

    try:
        # Stage 1: Download or Prepare Input Video
        progress(0.05, desc="[Stage 1/4] Preparing input video clip...")
        if youtube_url and youtube_url.strip():
            url = youtube_url.strip()
            trimmed_video, audio_path, meta = download_youtube_clip(
                url=url,
                start_time=start_time,
                duration=duration,
                output_dir=session["root"],
                max_height=max_dim,
                progress_callback=lambda p, msg: progress(0.05 + 0.20 * p, desc=f"[Stage 1/4] {msg}")
            )
            clip_title = meta.get("title", "YouTube Clip")
        else:
            # Process uploaded video
            ffmpeg_path = get_ffmpeg_path()
            input_file = Path(uploaded_video)
            trimmed_video = session["root"] / "trimmed_input.mp4"
            audio_path = session["audio"] / "original_audio.aac"

            import subprocess
            # Trim uploaded video
            subprocess.run([
                ffmpeg_path, "-y",
                "-ss", str(start_time),
                "-i", str(input_file),
                "-t", str(duration),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(trimmed_video)
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Extract audio
            subprocess.run([
                ffmpeg_path, "-y",
                "-i", str(trimmed_video),
                "-vn", "-c:a", "aac",
                str(audio_path)
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                audio_path = None
            clip_title = input_file.stem

        # Stage 2: Extract Frames
        progress(0.28, desc="[Stage 2/4] Extracting frames with OpenCV...")
        extraction = extract_frames(
            video_path=trimmed_video,
            output_dir=session["raw_frames"],
            max_dim=max_dim,
            progress_callback=lambda p, msg: progress(0.28 + 0.12 * p, desc=f"[Stage 2/4] {msg}")
        )

        fps = extraction.fps
        total_frames = extraction.frame_count

        # Stage 3: AI Stylization
        progress(0.42, desc=f"[Stage 3/4] Stylizing frames with {preset_name}...")
        stylized_paths = process_frames(
            frame_paths=extraction.frame_paths,
            output_dir=session["stylized_frames"],
            preset_name=preset_name,
            intensity=intensity,
            custom_prompt=custom_prompt,
            seed=42,
            smoothing_factor=0.15,
            progress_callback=lambda p, msg: progress(0.42 + 0.45 * p, desc=f"[Stage 3/4] {msg}")
        )

        # Stage 4: Reassemble & Audio Muxing
        progress(0.90, desc="[Stage 4/4] Encoding H.264 video & muxing audio...")
        timestamp = int(time.time())
        clean_preset = preset_name.replace(" ", "_").lower()
        output_filename = f"stylized_{clean_preset}_{timestamp}.mp4"
        final_output_path = OUTPUT_DIR / output_filename

        assemble_video(
            frames_dir=session["stylized_frames"],
            fps=fps,
            output_path=final_output_path,
            audio_path=audio_path,
            progress_callback=lambda p, msg: progress(0.90 + 0.10 * p, desc=f"[Stage 4/4] {msg}")
        )

        # Prepare summary status
        status_md = f"""
### Success! Animation Rendered
- **Title**: {clip_title}
- **Preset Applied**: {preset_name}
- **Resolution**: {extraction.width} x {extraction.height}
- **Processed Frames**: {total_frames} @ {fps:.1f} FPS
- **Audio Synced**: {"Yes (Original Audio Preserved)" if audio_path else "Silent"}
- **Saved To**: `{final_output_path.name}`
"""
        return str(trimmed_video), str(final_output_path), status_md, str(final_output_path)

    finally:
        # Automatic session cleanup of raw extracted PNG frames
        cleanup_session(session_root)


def build_app():
    # Setup custom theme styling
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    .badge-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 14px;
        color: #f8fafc;
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-cpu { background-color: #3b82f6; color: white; }
    .badge-gpu { background-color: #10b981; color: white; }
    """

    with gr.Blocks(title="AI Video Stylizer") as demo:
        gr.Markdown("# 🎬 Local YouTube to AI Animation Converter")

        # Hardware Status Banner
        gpu_badge_class = "badge-gpu" if HARDWARE.has_cuda else "badge-cpu"
        hardware_badge_text = f"Hardware: {HARDWARE.device_name}"
        if HARDWARE.has_cuda:
            hardware_badge_text += f" ({HARDWARE.vram_gb} GB VRAM)"
        else:
            hardware_badge_text += " | Fast Engine Active (AnimeGANv2 & Artistic Shaders)"

        gr.HTML(f"""
        <div class="badge-box">
            <span class="badge-tag {gpu_badge_class}">{hardware_badge_text}</span>
            <span style="margin-left: 12px; font-size: 0.9rem; color: #94a3b8;">
                {HARDWARE.warning_message}
            </span>
        </div>
        """)

        with gr.Row():
            # Left Column: Controls & Inputs
            with gr.Column(scale=5):
                with gr.Tab("YouTube URL"):
                    yt_input = gr.Textbox(
                        label="YouTube Video URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        value=""
                    )
                    # Quick example buttons
                    gr.Markdown("**Quick Test Examples:**")
                    with gr.Row():
                        ex1 = gr.Button("Nature Waterfall (CC)", size="sm")
                        ex2 = gr.Button("Synthwave Runner (CC)", size="sm")
                        ex3 = gr.Button("Tokyo Street Walk (CC)", size="sm")

                with gr.Tab("Upload Local Video"):
                    file_input = gr.Video(
                        label="Drag and Drop Video File",
                        sources=["upload"]
                    )

                with gr.Group():
                    gr.Markdown("### ✂️ Clip Trimmer (Seconds)")
                    with gr.Row():
                        start_slider = gr.Slider(
                            minimum=0.0,
                            maximum=600.0,
                            value=0.0,
                            step=0.5,
                            label="Start Time (sec)"
                        )
                        duration_slider = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=3.0,
                            step=0.5,
                            label="Clip Duration (sec)"
                        )

                with gr.Group():
                    gr.Markdown("### 🎨 AI Stylization Settings")
                    preset_choices = list(STYLE_PRESETS.keys())
                    preset_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value="Studio Ghibli Anime",
                        label="Animation Style Preset"
                    )
                    preset_desc = gr.Markdown(
                        f"*{STYLE_PRESETS['Studio Ghibli Anime']['description']}*"
                    )

                    with gr.Row():
                        intensity_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                            label="Style Conditioning Intensity"
                        )
                        resolution_dropdown = gr.Dropdown(
                            choices=["360p", "480p", "720p"],
                            value="480p",
                            label="Processing Resolution"
                        )

                    custom_prompt = gr.Textbox(
                        label="Custom Prompt Override (Optional)",
                        placeholder="Leave blank to use optimized preset prompt...",
                        lines=2
                    )

                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate AI Animation",
                        variant="primary",
                        size="lg"
                    )
                    clear_cache_btn = gr.Button(
                        "🧹 Clear Temp Cache",
                        variant="secondary",
                        size="lg"
                    )

            # Right Column: Live Output & Previews
            with gr.Column(scale=6):
                status_box = gr.Markdown("### Ready\n*Select a video clip and click 'Generate AI Animation' to begin.*")

                with gr.Row():
                    orig_preview = gr.Video(
                        label="Original Trimmed Clip",
                        interactive=False
                    )
                    stylized_preview = gr.Video(
                        label="✨ Stylized AI Animation (With Audio)",
                        interactive=False
                    )

                download_file = gr.File(
                    label="Download Final MP4",
                    interactive=False
                )

        # Event Handlers
        def update_desc(preset_name):
            desc = STYLE_PRESETS.get(preset_name, {}).get("description", "")
            return f"*{desc}*"

        preset_dropdown.change(
            fn=update_desc,
            inputs=[preset_dropdown],
            outputs=[preset_desc]
        )

        # Example clicks
        ex1.click(lambda: ("https://www.youtube.com/watch?v=aqz-KE-bpKQ", 10.0, 3.0), outputs=[yt_input, start_slider, duration_slider])
        ex2.click(lambda: ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 1.0, 3.0), outputs=[yt_input, start_slider, duration_slider])
        ex3.click(lambda: ("https://www.youtube.com/watch?v=1La4QzGeaaQ", 20.0, 3.0), outputs=[yt_input, start_slider, duration_slider])

        generate_btn.click(
            fn=run_pipeline,
            inputs=[
                yt_input,
                file_input,
                start_slider,
                duration_slider,
                preset_dropdown,
                intensity_slider,
                custom_prompt,
                resolution_dropdown
            ],
            outputs=[orig_preview, stylized_preview, status_box, download_file]
        )

        def clear_temp():
            cleanup_all_temp()
            return "🧹 Stale temporary frame directories and caches cleared."

        clear_cache_btn.click(
            fn=clear_temp,
            inputs=[],
            outputs=[status_box]
        )

    return demo, custom_css


if __name__ == "__main__":
    demo, css_rules = build_app()
    server_port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("HOST", "0.0.0.0")
    
    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "share": False,
    }
    import inspect
    sig = inspect.signature(demo.launch).parameters
    if "ssr_mode" in sig:
        launch_kwargs["ssr_mode"] = False
    if "theme" in sig:
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="indigo")
    if "css" in sig:
        launch_kwargs["css"] = css_rules

    demo.launch(**launch_kwargs)
