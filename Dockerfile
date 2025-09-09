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

# Python dependencies - staged installation for better reliability
COPY requirements.txt .

# Stage 1: Essential build tools and PyTorch
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install wheel ninja packaging setuptools

# Stage 2: PyTorch with CUDA support
RUN python3.10 -m pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Stage 3: Core ML libraries
RUN python3.10 -m pip install transformers>=4.41.0 diffusers>=0.30.0 accelerate>=0.34.0 huggingface_hub>=0.24.0

# Stage 4: Heavy optimization packages - using precompiled wheels where possible
RUN python3.10 -m pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases --no-build-isolation || \
    python3.10 -m pip install flash_attn --timeout 3600 --no-cache-dir
RUN python3.10 -m pip install xformers --extra-index-url https://download.pytorch.org/whl/cu121 || \
    python3.10 -m pip install xformers --timeout 1800 --no-cache-dir
RUN python3.10 -m pip install deepspeed fairscale

# Stage 5: Remaining packages
RUN python3.10 -m pip install opencv-python pillow numpy requests httpx aiofiles safetensors omegaconf einops runpod && \
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