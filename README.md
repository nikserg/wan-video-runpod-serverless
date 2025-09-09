[![Runpod](https://api.runpod.io/badge/nikserg/wan-video-runpod-serverless)](https://console.runpod.io/hub/nikserg/wan-video-runpod-serverless)

# WAN 2.2 Video Generation Serverless

RunPod serverless endpoint for WAN 2.2 video generation supporting both text-to-video (T2V) and image-to-video (I2V) generation with configurable duration and FPS.

## Features

- 🎥 **Text-to-Video (T2V)** - Generate videos from text prompts
- 🖼️ **Image-to-Video (I2V)** - Animate static images with text guidance  
- ⏱️ **Configurable Duration** - Set video length in seconds
- 🎬 **Adjustable FPS** - Control frame rate (1-60 FPS)
- 📦 **Base64 MP4 Output** - Ready-to-use video format
- 🎨 **LoRA Support** - Load custom LoRAs from CivitAI
- ⚡ **GPU Optimized** - Supports consumer and enterprise GPUs
- 🔄 **Auto Model Download** - Models cached in persistent volumes

## Quick Start

### Deploy to RunPod

1. Click the RunPod badge above or visit the [Hub page](https://console.runpod.io/hub/nikserg/wan-video-runpod-serverless)
2. Choose a preset based on your GPU:
   - **TI2V-5B** - RTX 4090+ (24GB VRAM)
   - **T2V-A14B** - A100+ (80GB VRAM) 
   - **I2V-A14B** - A100+ (80GB VRAM)
3. Configure LoRA URLs if needed
4. Deploy and wait for model initialization

### API Usage

```python
import requests
import base64

# Text-to-Video
response = requests.post("YOUR_ENDPOINT_URL", json={
    "input": {
        "prompt": "A beautiful sunset over the ocean with waves crashing",
        "duration_seconds": 5.0,
        "fps": 24,
        "width": 1280,
        "height": 720
    }
})

# Image-to-Video  
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post("YOUR_ENDPOINT_URL", json={
    "input": {
        "prompt": "Camera slowly pans across the landscape", 
        "image": image_b64,
        "duration_seconds": 3.0,
        "fps": 16
    }
})

# Save generated video
video_data = base64.b64decode(response.json()["video_base64"])
with open("generated_video.mp4", "wb") as f:
    f.write(video_data)
```

## API Reference

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | ✅ | - | Text description for video generation |
| `image` | string | ❌ | - | Base64 encoded input image for I2V |
| `duration_seconds` | float | ❌ | 5.0 | Video length in seconds |
| `fps` | integer | ❌ | 24 | Frames per second (1-60) |
| `width` | integer | ❌ | 1280 | Video width in pixels |
| `height` | integer | ❌ | 720 | Video height in pixels |
| `negative_prompt` | string | ❌ | "" | What to avoid in generation |
| `guidance_scale` | float | ❌ | 5.0 | How closely to follow prompt |
| `num_inference_steps` | integer | ❌ | 50 | Quality vs speed tradeoff |
| `seed` | integer | ❌ | random | Random seed for reproducibility |

### Response Format

```json
{
  "video_base64": "UklGRiQAAABXRUJQVlA4...",
  "prompt": "A beautiful sunset...",
  "duration_seconds": 5.0,
  "fps": 24,
  "resolution": "1280x720",
  "has_input_image": false,
  "model_type": "TI2V-5B"
}
```

## Environment Variables

### Model Configuration
- `WAN_MODEL_TYPE` - Model variant (TI2V-5B, T2V-A14B, I2V-A14B)
- `USE_MOCK_GENERATOR` - Enable mock generator for testing (true/false)

### LoRA Configuration
- `LORA_URL_1` - First LoRA download URL (CivitAI supported)
- `LORA_URL_2` - Second LoRA URL (optional)
- `LORA_URL_3` - Third LoRA URL (optional)

## WAN 2.2 Models

### TI2V-5B (Recommended)
- **GPU**: RTX 4090+ (24GB VRAM)
- **Features**: Text-to-video + Image-to-video
- **Performance**: ~9 minutes for 720p video
- **Best for**: Consumer GPU setups

### T2V-A14B (High-end)
- **GPU**: A100+ (80GB VRAM)
- **Features**: Text-to-video only
- **Performance**: Superior quality
- **Best for**: Enterprise deployments

### I2V-A14B (High-end)
- **GPU**: A100+ (80GB VRAM)  
- **Features**: Image-to-video only
- **Performance**: Superior quality
- **Best for**: Professional I2V workflows

## Hardware Requirements

| Model | Min VRAM | Recommended GPU | Performance |
|-------|----------|-----------------|-------------|
| TI2V-5B | 24GB | RTX 4090 | 720p in ~9min |
| T2V-A14B | 80GB | A100/H100 | High quality |
| I2V-A14B | 80GB | A100/H100 | High quality |

**Storage**: 100GB+ container disk for models and cache

## Local Development

```bash
# Clone repository
git clone https://github.com/nikserg/wan-video-runpod-serverless.git
cd wan-video-runpod-serverless

# Build Docker image
docker build -t wan-video-serverless .

# Run with mock generator (for testing)
docker run --rm -p 8000:8000 \
  -e USE_MOCK_GENERATOR=true \
  wan-video-serverless

# Run with real model (requires GPU)
docker run --rm -p 8000:8000 --gpus all \
  -e WAN_MODEL_TYPE=TI2V-5B \
  -e USE_MOCK_GENERATOR=false \
  wan-video-serverless
```

## Examples

### Text-to-Video
```json
{
  "input": {
    "prompt": "A cat wearing sunglasses riding a skateboard in slow motion",
    "duration_seconds": 4.0,
    "fps": 24,
    "width": 1280,
    "height": 720
  }
}
```

### Image-to-Video with Custom Settings
```json
{
  "input": {
    "prompt": "The flowers gently sway in the wind",
    "image": "base64_encoded_flower_image",
    "duration_seconds": 6.0,
    "fps": 30,
    "negative_prompt": "static, still, frozen",
    "guidance_scale": 7.0,
    "seed": 12345
  }
}
```

## Troubleshooting

### Common Issues

**Model Download Fails**
- Check internet connectivity
- Verify Hugging Face access
- Ensure sufficient disk space (100GB+)

**Out of Memory Errors**  
- Use TI2V-5B model for consumer GPUs
- Enable `USE_MOCK_GENERATOR=true` for testing
- Reduce video resolution or duration

**Slow Generation**
- Lower `num_inference_steps` (e.g., 30)
- Use smaller resolution (720p → 480p)
- Reduce duration or FPS

### Mock Generator Mode
For testing without GPU requirements:
```bash
docker run -e USE_MOCK_GENERATOR=true wan-video-serverless
```

## License

Apache 2.0 License - see [WAN 2.2 repository](https://github.com/Wan-Video/Wan2.2) for model license details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with mock generator
5. Submit a pull request

## Support

- 🐛 [Report Issues](https://github.com/nikserg/wan-video-runpod-serverless/issues)
- 💬 [RunPod Community](https://discord.gg/runpod)