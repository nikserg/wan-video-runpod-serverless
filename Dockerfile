# Multi-stage build for smaller final image
FROM runpod/base:0.7.0-cuda1290-ubuntu2404 AS builder

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create directories
RUN mkdir -p /models /loras /workspace

# Install PyTorch 2.8 first to match flash_attn wheel
RUN python3.10 -m pip install torch>=2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install prebuilt flash_attn wheel (compiled for torch 2.8)
RUN python3.10 -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.8-cp310-cp310-linux_x86_64.whl

# Install Wan2.2 from source (dependencies already satisfied)
RUN cd /workspace && \
    git clone https://github.com/Wan-Video/Wan2.2.git && \
    cd Wan2.2 && \
    python3.10 -m pip install -e . && \
    rm -rf .git

# Final stage
FROM runpod/base:0.7.0-cuda1290-ubuntu2404

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create directories
RUN mkdir -p /models /loras /workspace

# Copy Wan2.2 and all Python packages from builder stage
COPY --from=builder /workspace/Wan2.2 /workspace/Wan2.2
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install only additional packages not installed by WAN 2.2
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r /tmp/requirements.txt && \
    rm -rf ~/.cache/pip /tmp/* /var/tmp/* /tmp/requirements.txt

# Copy application files
COPY rp_handler.py /rp_handler.py
COPY model_downloader.py /model_downloader.py
COPY video_generator.py /video_generator.py

WORKDIR /workspace/Wan2.2

CMD python3.10 -u /rp_handler.py