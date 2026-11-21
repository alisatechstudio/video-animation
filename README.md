---
title: Video Animation
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# 🎬 Local & Cloud YouTube to AI Animation Converter

A modular web application that takes a YouTube video URL (or a local video file), transforms its frames into stylized AI animation, and reassembles the final video with the original audio track synchronized.

---

## ☁️ Cloud Deployment from GitHub (Important Notice)

### Can I deploy this directly to Netlify?
> [!WARNING]
> **Netlify is designed for static websites and short-lived serverless functions (10-second timeout, 250MB size limit).**
> It cannot run persistent Python web servers (`app.py`), long-running video downloads (`yt-dlp`), or heavy computer vision neural frame processing (`OpenCV` / `ONNX` / `PyTorch`), which require persistent execution, system FFmpeg, and several seconds/minutes of continuous compute.

### What are the Best 100% Free Cloud Alternatives to Netlify?

#### Option 1: Hugging Face Spaces (Recommended — 100% Free & Built for Gradio)
Hugging Face Spaces is designed specifically for Gradio apps and gives you **16 GB RAM, 2 vCPUs, and 50 GB storage for free**:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space SDK: Select **Gradio**.
3. Under **Space Hardware**, select **CPU Basic (Free)**.
4. Clone the space or connect your GitHub repository using GitHub Actions.
5. Hugging Face automatically detects `requirements.txt`, `packages.txt` (which installs FFmpeg), and runs `app.py`.

#### Option 2: Render.com / Railway / Koyeb (Docker Web Service)
1. Push this codebase to your GitHub repository.
2. In [Render](https://render.com) or [Railway](https://railway.app), create a **New Web Service** and link your GitHub repo.
3. Select **Docker** environment (the included `Dockerfile` will automatically install FFmpeg, configure dependencies, and expose port 7860).
4. Click **Deploy**.

#### Option 3: Netlify Frontend + Cloud Backend (Hybrid Architecture)
If you specifically need a `.netlify.app` web address:
1. Deploy the Python backend on Hugging Face Spaces or Render for free.
2. Deploy a lightweight HTML/JS or Next.js frontend to Netlify that sends API requests to your cloud backend's Gradio API client (`@gradio/client`).

---

## 🐙 Push This Project to GitHub

Run these commands in PowerShell or Terminal to initialize and push your repository to GitHub:

```powershell
# 1. Initialize git
git init

# 2. Add files
git add .

# 3. Create initial commit
git commit -m "feat: complete local & cloud YouTube AI animation web app"

# 4. Link your GitHub repository (replace with your repo URL)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 5. Push to GitHub
git push -u origin main
```

---

## 🎨 Style Presets Included

1. **Studio Ghibli Anime**: AnimeGANv2 (Hayao weight) ONNX model; lush anime backgrounds, soft natural lighting, and warm pastoral aesthetics.
2. **Makoto Shinkai Anime**: AnimeGANv2 (Shinkai weight) ONNX model; high vibrance, dramatic cinematic sky tones, and intense emotional lighting.
3. **Cyberpunk Neon**: Dynamic Canny edge trace with neon contour glow dilation, deep blue/cyan grading, and synthwave magenta highlights.
4. **Watercolor Painting**: Multi-pass bilateral diffusion, delicate edge thresholding, and pigment quantization on paper grain.
5. **Comic Book Ink**: Adaptive edge linework, bold ink contours, and cel-shaded color posterization.
6. **Custom SD ControlNet Canny**: Full generative diffusion via `runwayml/stable-diffusion-v1-5` and `lllyasviel/sd-controlnet-canny` (recommended for CUDA GPUs with ≥8GB VRAM).

---

## 📂 Project Architecture

```
├── Dockerfile                     # Cloud container deployment (Render / Railway / Koyeb)
├── packages.txt                   # System packages for Hugging Face Spaces (FFmpeg)
├── requirements.txt               # Lightweight cloud & CPU requirements
├── requirements-gpu.txt           # Optional full GPU Stable Diffusion requirements
├── app.py                         # Modern Gradio web user interface (port 7860, 0.0.0.0)
├── test_pipeline.py               # Automated synthetic video CLI smoke test
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hardware discovery & style preset definitions
│   ├── utils.py                   # FFmpeg auto-discovery, session management & cleanup
│   ├── downloader.py              # yt-dlp downloader with section trimming & audio extraction
│   ├── extractor.py               # cv2 frame extractor with resolution scaling
│   ├── animator.py                # Dual stylization engine (AnimeGANv2 + CV shaders + SD ControlNet)
│   └── assembler.py               # FFmpeg H.264 video assembler & AAC audio muxer
└── .gitignore                     # Prevents large temp files & venv from entering git
```

---

## 💻 Local Quick Start (If running locally)

```powershell
# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run automated smoke test
python test_pipeline.py

# Launch web app
python app.py
```
Open **`http://localhost:7860`** in your browser.
