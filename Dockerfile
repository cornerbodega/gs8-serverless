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
    flash-attn --no-build-isolation && \
    pip install --no-cache-dir transformers==4.41.2

# Python deps for RVM
RUN pip install --no-cache-dir av pims

# RunPod serverless SDK
RUN pip install --no-cache-dir runpod

# Clone HVA
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /app/hva

# Download HVA weights (~30GB)
RUN pip install --no-cache-dir huggingface_hub && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/HunyuanVideo-Avatar', local_dir='/app/hva/weights')"

# Clone RVM + download weights
RUN git clone https://github.com/PeterL1n/RobustVideoMatting.git /app/rvm && \
    python3 -c "import torch; model = torch.hub.load('PeterL1n/RobustVideoMatting', 'resnet50'); torch.save(model.state_dict(), '/app/rvm/rvm_resnet50.pth')"

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python3", "-u", "/app/handler.py"]
