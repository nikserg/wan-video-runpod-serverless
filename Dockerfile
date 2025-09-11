FROM runpod/base:0.7.0-ubuntu2204-cuda1241

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create directories
RUN mkdir -p /workspace

RUN python3.10 -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.5-cp310-cp310-linux_x86_64.whl

RUN python3.10 -m pip install torchvision --index-url https://download.pytorch.org/whl/cu124


# Install additional requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Clone and install Wan2.2
RUN cd /workspace && \
    git clone https://github.com/Wan-Video/Wan2.2.git && \
    cd Wan2.2 && \
    python3.10 -m pip install -e . && \
    rm -rf .git

# Copy application files
COPY rp_handler.py /rp_handler.py
COPY model_downloader.py /model_downloader.py
COPY video_generator.py /video_generator.py

WORKDIR /workspace/Wan2.2

CMD python3.10 -u /rp_handler.py