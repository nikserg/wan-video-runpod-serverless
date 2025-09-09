import os
import io
import base64
import logging

import runpod
import requests
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np
import imageio
import torch

logging.basicConfig(level=logging.INFO)

MODEL_REPO = os.environ.get("WAN_MODEL_REPO", "Wan-Labs/Wan2.2-Video")
MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/workspace/wan2.2")
LORA_URLS = os.environ.get("WAN_LORA_URLS", "")
LORA_DIR = os.path.join(MODEL_DIR, "loras")

def download_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "model_index.json")):
        logging.info("Downloading WAN 2.2 model to %s", MODEL_DIR)
        snapshot_download(MODEL_REPO, local_dir=MODEL_DIR, local_dir_use_symlinks=False)

def download_loras():
    if not LORA_URLS:
        return
    os.makedirs(LORA_DIR, exist_ok=True)
    for url in [u.strip() for u in LORA_URLS.split(",") if u.strip()]:
        filename = os.path.join(LORA_DIR, os.path.basename(url.split("?")[0]))
        if os.path.exists(filename):
            continue
        logging.info("Downloading LoRA from %s", url)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

def load_loras(pipeline):
    if not os.path.isdir(LORA_DIR):
        return
    for name in os.listdir(LORA_DIR):
        if name.endswith(".safetensors") or name.endswith(".pt"):
            try:
                pipeline.load_lora_weights(LORA_DIR, weight_name=name)
                logging.info("Loaded LoRA %s", name)
            except Exception as e:
                logging.warning("Failed to load LoRA %s: %s", name, e)

download_model()
download_loras()
PIPELINE = DiffusionPipeline.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16
).to("cuda")
load_loras(PIPELINE)

def generate_video(prompt: str, image_b64: str | None, num_frames: int, fps: int) -> str:
    if image_b64:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        output = PIPELINE(prompt=prompt, image=image, num_frames=num_frames)
    else:
        output = PIPELINE(prompt=prompt, num_frames=num_frames)

    frames = output.frames if hasattr(output, "frames") else output.images
    if isinstance(frames[0], Image.Image):
        frames = [np.array(f) for f in frames]

    buffer = io.BytesIO()
    imageio.mimwrite(buffer, frames, format="mp4", fps=fps)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def handler(job):
    data = job["input"]
    prompt = data.get("prompt", "")
    image_b64 = data.get("image")
    num_frames = int(data.get("num_frames", 16))
    fps = int(data.get("fps", 24))

    video_b64 = generate_video(prompt, image_b64, num_frames, fps)
    return {"video": video_b64, "num_frames": num_frames, "fps": fps}

runpod.serverless.start({"handler": handler})
