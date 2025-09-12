FROM runpod/base:0.7.0-ubuntu2204-cuda1281

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libgl1-mesa-dev libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for HuggingFace large file transfers (optional)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN mkdir -p /app

# --- Install PyTorch 2.7 (CUDA 12.8) and torchvision ---
RUN python3.10 -m pip install --upgrade pip && \
    pip install \
      torch==2.8.0+cu128 \
      torchvision==0.23.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

# --- Install FlashAttention from prebuilt wheel (compatible with torch 2.7) ---
RUN python3.10 -m pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy and install Python dependencies (excluding flash_attn to avoid rebuild)
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt


# Copy application files
COPY ./app /app
COPY ./wan /wan

WORKDIR /app
CMD ["python3.10", "-u", "/app/rp_handler.py"]
