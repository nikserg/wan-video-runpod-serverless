FROM runpod/base:0.6.1-cuda12.2.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE

# Python dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install wheel && \
    python3.10 -m pip install ninja && \
    python3.10 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Create directories for models and LoRAs
RUN mkdir -p /models /loras /workspace

# Clone WAN 2.2 repository
RUN cd /workspace && \
    git clone https://github.com/Wan-Video/Wan2.2.git && \
    cd Wan2.2 && \
    python3.10 -m pip install -e .

# Copy handler and utility files
COPY rp_handler.py /rp_handler.py
COPY model_downloader.py /model_downloader.py
COPY video_generator.py /video_generator.py

# Set working directory
WORKDIR /workspace/Wan2.2

CMD python3.10 -u /rp_handler.py