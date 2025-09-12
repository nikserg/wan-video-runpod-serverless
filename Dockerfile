FROM runpod/base:0.7.0-ubuntu2204-cuda1281

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libgl1-mesa-dev libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for HuggingFace large file transfers (optional)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create workspace directory
RUN mkdir -p /workspace

# --- Install PyTorch 2.7 (CUDA 12.8) and torchvision ---
RUN python3.10 -m pip install --upgrade pip && \
    pip install \
      torch==2.7.0+cu128 \
      torchvision==0.22.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

# --- Install FlashAttention from prebuilt wheel (compatible with torch 2.7) ---
RUN python3.10 -m pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy and install Python dependencies (excluding flash_attn to avoid rebuild)
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Clone and install Wan2.2
RUN git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2 && \
    python3.10 -m pip install -e /workspace/Wan2.2 && \
    rm -rf /workspace/Wan2.2/.git

# Show all packages installed
RUN python3.10 -m pip list

# Copy application files
COPY rp_handler.py /rp_handler.py
COPY model_downloader.py /model_downloader.py
COPY video_generator.py /video_generator.py
COPY check_cuda.py /check_cuda.py

WORKDIR /workspace/Wan2.2
CMD ["python3.10", "-u", "/rp_handler.py"]
