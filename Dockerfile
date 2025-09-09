FROM runpod/base:0.7.0-cuda1290-ubuntu2404


RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE

COPY requirements.txt .

RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install wheel ninja packaging setuptools


RUN python3.10 -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.8-cp310-cp310-linux_x86_64.whl

RUN python3.10 -m pip install torch>=2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


RUN python3.10 -m pip install transformers>=4.41.0 diffusers>=0.30.0 accelerate>=0.34.0 huggingface_hub>=0.24.0


RUN python3.10 -m pip install xformers --extra-index-url https://download.pytorch.org/whl/cu121


RUN python3.10 -m pip install deepspeed fairscale


RUN python3.10 -m pip install opencv-python pillow numpy requests httpx aiofiles safetensors omegaconf einops runpod && \
    rm /requirements.txt

RUN mkdir -p /models /loras /workspace

RUN cd /workspace && \
    git clone https://github.com/Wan-Video/Wan2.2.git && \
    cd Wan2.2 && \
    python3.10 -m pip install -e .

COPY rp_handler.py /rp_handler.py
COPY model_downloader.py /model_downloader.py
COPY video_generator.py /video_generator.py

WORKDIR /workspace/Wan2.2

CMD python3.10 -u /rp_handler.py