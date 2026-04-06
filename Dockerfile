FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps for HVA
RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    diffusers==0.33.0 \
    accelerate==1.1.1 \
    opencv-python==4.9.0.80 \
    pandas==2.0.3 \
    numpy==1.24.4 \
    einops==0.7.0 \
    tqdm==4.66.2 \
    loguru==0.7.2 \
    imageio==2.34.0 \
    imageio-ffmpeg==0.5.1 \
    safetensors==0.4.3 \
    decord==0.6.0 \
    librosa==0.11.0 \
    scikit-video==1.1.11 \
    huggingface_hub && \
    pip install --no-cache-dir transformers==4.41.2

# flash-attn needs CUDA at build time — skip if no GPU, install at runtime
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn skipped (no GPU at build time)"

# Python deps for RVM
RUN pip install --no-cache-dir av pims

# RunPod serverless SDK
RUN pip install --no-cache-dir runpod

# Clone HVA code (no weights — downloaded at runtime to /runpod-volume)
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /app/hva

# Clone RVM code
RUN git clone https://github.com/PeterL1n/RobustVideoMatting.git /app/rvm

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python3", "-u", "/app/handler.py"]
