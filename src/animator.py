"""
Animation and Stylization Engine.
Supports ultra-fast ONNX AnimeGANv2, OpenCV Artistic Shaders, and SD ControlNet Canny.
"""
from pathlib import Path
from typing import List, Optional, Any, Tuple
import cv2
import numpy as np
from PIL import Image

from src.config import detect_hardware, STYLE_PRESETS, MODELS_DIR
from src.utils import ensure_onnx_model


class ArtisticShaders:
    """
    High-performance Computer Vision Stylizers running at >30 FPS on CPU.
    """

    @staticmethod
    def cyberpunk(img_bgr: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Cyberpunk neon aesthetic with glowing cyan/magenta contours."""
        h, w = img_bgr.shape[:2]
        # Smooth and boost contrast
        smooth = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

        # Canny edge detection for neon trace
        edges = cv2.Canny(gray, 60, 140)
        glow = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        glow_blur = cv2.GaussianBlur(glow, (9, 9), 0)

        # Color grading: shift shadows to deep cyan, highlights to magenta
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        # Shift green-red channel towards magenta and blue-yellow towards deep blue/cyan
        a = np.clip(a + 25 * intensity, 0, 255)
        b = np.clip(b - 30 * intensity, 0, 255)
        graded_lab = cv2.merge([l, a, b]).astype(np.uint8)
        base = cv2.cvtColor(graded_lab, cv2.COLOR_LAB2BGR)

        # Generate neon edge masks
        cyan_neon = np.zeros_like(img_bgr)
        cyan_neon[:, :, 0] = glow_blur  # Blue
        cyan_neon[:, :, 1] = glow_blur  # Green
        cyan_neon[:, :, 2] = glow_blur // 4  # Low red

        magenta_neon = np.zeros_like(img_bgr)
        magenta_neon[:, :, 0] = glow  # Blue
        magenta_neon[:, :, 1] = 0
        magenta_neon[:, :, 2] = glow  # Red

        # Combine
        combined = cv2.addWeighted(base, 0.7, cyan_neon, 0.6 * intensity, 0)
        combined = cv2.addWeighted(combined, 1.0, magenta_neon, 0.7 * intensity, 0)
        return np.clip(combined, 0, 255).astype(np.uint8)

    @staticmethod
    def watercolor(img_bgr: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Multi-layer watercolor painting effect with edge pigment bleeding."""
        # Multi-pass bilateral filter to simulate soft water diffusion
        diffused = img_bgr
        passes = 3 if intensity > 0.6 else 2
        for _ in range(passes):
            diffused = cv2.bilateralFilter(diffused, d=9, sigmaColor=90, sigmaSpace=90)

        # Delicate edge lines
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            blurred_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
        )
        edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Color quantization for pigment clustering
        data = np.float32(diffused.reshape((-1, 3)))
        k = 12
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(diffused.shape)

        # Multiply pigment with edge canvas
        watercolor = cv2.bitwise_and(quantized, edge_rgb)

        # Warm paper texture tone
        paper_overlay = np.full_like(img_bgr, (245, 250, 255), dtype=np.uint8)
        result = cv2.addWeighted(watercolor, 0.85, paper_overlay, 0.15, 0)
        return result

    @staticmethod
    def comic(img_bgr: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Retro comic book style with bold ink lineart and cel-shaded color blocks."""
        # 1. Bold ink lines
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=4
        )
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 2. Cel shading color posterization
        div = 64
        cel = (img_bgr // div) * div + div // 2

        # 3. Enhance saturation
        hsv = cv2.cvtColor(cel, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.2 + 0.3 * intensity), 0, 255)
        cel_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Combine ink with cel colors
        comic_frame = cv2.bitwise_and(cel_boosted, edge_bgr)
        return comic_frame


class AnimeGANv2Runner:
    """
    Lightweight ONNX inference for AnimeGANv2 (Hayao / Shinkai).
    """
    def __init__(self, variant: str = "hayao"):
        self.variant = variant
        self.session = None
        self._init_session()

    def _init_session(self):
        try:
            import onnxruntime as ort
            model_file = f"AnimeGANv2_{self.variant.capitalize()}.onnx"
            model_path = ensure_onnx_model(model_file)
            if model_path and model_path.exists():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self.session = ort.InferenceSession(str(model_path), opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"Could not load AnimeGANv2 ONNX session: {e}. Will fallback to artistic shader.")
            self.session = None

    def stylize(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            # High-quality fallback to artistic pastoral anime grade
            return ArtisticShaders.watercolor(frame_bgr, intensity=0.9)

        orig_h, orig_w = frame_bgr.shape[:2]
        # Target 512px height preserving aspect ratio
        target_h, target_w = 512, int(512 * (orig_w / orig_h))
        # Round to multiple of 32
        target_w = (target_w // 32) * 32
        target_h = 512

        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB and normalize to [-1, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 127.5) - 1.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :]

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: tensor})
        out_tensor = outputs[0][0]

        # Postprocess from [-1, 1] to [0, 255]
        out_img = np.transpose(out_tensor, (1, 2, 0))
        out_img = (out_img + 1.0) * 127.5
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        # Resize back to original
        return cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)


class SDControlNetRunner:
    """
    Stable Diffusion ControlNet Canny pipeline for systems with >= 8GB VRAM or explicit GPU execution.
    """
    def __init__(self):
        self.pipe = None
        self._init_pipeline()

    def _init_pipeline(self):
        import torch
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

        hardware = detect_hardware()
        dtype = torch.float16 if hardware.has_cuda else torch.float32
        device = "cuda" if hardware.has_cuda else "cpu"

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if hardware.has_cuda:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cpu")

    def stylize_frame(
        self,
        frame_bgr: np.ndarray,
        positive_prompt: str,
        negative_prompt: str,
        conditioning_scale: float = 0.8,
        seed: int = 42
    ) -> np.ndarray:
        import torch
        # 1. Extract Canny edges
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        canny_3ch = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(canny_3ch)

        # 2. Fixed generator for temporal consistency
        generator = torch.manual_seed(seed)

        # 3. Run inference
        result = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=20,
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator
        ).images[0]

        res_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return cv2.resize(res_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))


def process_frames(
    frame_paths: List[Path],
    output_dir: Path,
    preset_name: str,
    intensity: float = 0.8,
    custom_prompt: Optional[str] = None,
    seed: int = 42,
    smoothing_factor: float = 0.15,
    progress_callback: Optional[Any] = None
) -> List[Path]:
    """
    Sequentially stylizes all video frames using the selected preset with temporal smoothing.

    Args:
        frame_paths: List of input frame paths.
        output_dir: Directory where stylized frames will be stored.
        preset_name: Selected style preset name.
        intensity: Conditioning strength / stylization intensity (0.1 - 1.0).
        custom_prompt: Optional prompt override.
        seed: Deterministic random seed to minimize temporal flickering.
        smoothing_factor: Exponential moving average blend with previous frame (0.0 = none, 0.2 = smooth).
        progress_callback: Optional progress reporter callable.

    Returns:
        List of paths to stylized PNG frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = STYLE_PRESETS.get(preset_name, STYLE_PRESETS["Studio Ghibli Anime"])
    engine = preset["engine"]

    # Initialize runner based on engine
    animegan_runner = None
    diffusion_runner = None

    if engine == "animegan2":
        animegan_runner = AnimeGANv2Runner(variant=preset.get("model_variant", "hayao"))
    elif engine == "diffusion":
        diffusion_runner = SDControlNetRunner()

    stylized_paths: List[Path] = []
    prev_stylized: Optional[np.ndarray] = None
    total = len(frame_paths)

    for i, f_path in enumerate(frame_paths):
        frame = cv2.imread(str(f_path))
        if frame is None:
            continue

        # Stylize according to active engine
        if engine == "animegan2":
            stylized = animegan_runner.stylize(frame)
        elif engine == "artistic":
            shader = preset.get("shader", "cyberpunk")
            if shader == "cyberpunk":
                stylized = ArtisticShaders.cyberpunk(frame, intensity)
            elif shader == "watercolor":
                stylized = ArtisticShaders.watercolor(frame, intensity)
            elif shader == "comic":
                stylized = ArtisticShaders.comic(frame, intensity)
            else:
                stylized = ArtisticShaders.watercolor(frame, intensity)
        elif engine == "diffusion":
            prompt = custom_prompt if custom_prompt else preset["positive_prompt"]
            stylized = diffusion_runner.stylize_frame(
                frame,
                positive_prompt=prompt,
                negative_prompt=preset["negative_prompt"],
                conditioning_scale=intensity,
                seed=seed
            )
        else:
            stylized = frame

        # Temporal Consistency Smoothing: Blend slightly with previous frame
        if prev_stylized is not None and smoothing_factor > 0:
            stylized = cv2.addWeighted(
                stylized, 1.0 - smoothing_factor,
                prev_stylized, smoothing_factor,
                0
            )

        prev_stylized = stylized.copy()

        out_path = output_dir / f"stylized_{i:05d}.png"
        cv2.imwrite(str(out_path), stylized)
        stylized_paths.append(out_path)

        if progress_callback:
            progress_callback(
                (i + 1) / total,
                f"Stylizing frame {i + 1}/{total} ({preset_name})..."
            )

    return stylized_paths
