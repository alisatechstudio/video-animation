# Universal Cloud Deployment Dockerfile (Render, Railway, Koyeb, Hugging Face)
FROM python:3.11-slim

# Install system dependencies including FFmpeg and graphic libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure cache/output directories exist with proper write permissions
RUN mkdir -p /app/models /app/temp /app/outputs && \
    chmod -R 777 /app/models /app/temp /app/outputs

# Expose default Gradio port
ENV PORT=7860
ENV HOST=0.0.0.0
EXPOSE 7860

CMD ["python", "-u", "app.py"]
