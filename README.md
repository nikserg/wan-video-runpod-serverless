[![Runpod](https://api.runpod.io/badge/nikserg/wan-video-runpod-serverless)](https://console.runpod.io/hub/nikserg/wan-video-runpod-serverless)

# WAN Video RunPod Serverless

Serverless handler for generating video with the **WAN 2.2** model. On cold start the container downloads the model into the attached volume and applies optional LoRA weights specified via environment variables. The endpoint supports both text-to-video (T2V) and image-to-video (I2V) generation.

## Environment variables
- `WAN_MODEL_REPO` – Hugging Face repository containing the WAN model. Defaults to `Wan-Labs/Wan2.2-Video`.
- `WAN_MODEL_DIR` – Directory in the mounted volume where the model will be stored. Defaults to `/workspace/wan2.2`.
- `WAN_LORA_URLS` – Comma separated list of URLs to LoRA weights to download and load at startup.

## Request format
```
{
  "prompt": "text prompt",
  "image": "<base64 image> // optional",
  "num_frames": 16,
  "fps": 24
}
```

The handler returns a dictionary containing a `video` field with the base64 encoded MP4, alongside the frame count and FPS used.
