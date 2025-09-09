[![Runpod](https://api.runpod.io/badge/nikserg/wan-video-runpod-serverless)](https://console.runpod.io/hub/nikserg/wan-video-runpod-serverless)

# WAN 2.2 Video Generation Serverless

RunPod serverless endpoint for WAN 2.2 video generation supporting both text-to-video (T2V) and image-to-video (I2V) generation with configurable duration and FPS.

## Features

- 🎥 **Text-to-Video (T2V)** - Generate videos from text prompts
- 🖼️ **Image-to-Video (I2V)** - Animate static images with text guidance  
- 📐 **Resolution Presets** - 480p, 720p, vertical, square formats
- ⏱️ **Configurable Duration** - Set video length in seconds
- 🎬 **Adjustable FPS** - Control frame rate (1-60 FPS)
- 📦 **Base64 MP4 Output** - Ready-to-use video format
- 🎨 **LoRA Support** - Load custom LoRAs from CivitAI
- 🔧 **Multiple Models** - From 3.5GB to 80GB VRAM options
- ⚡ **GPU Optimized** - Supports consumer and enterprise GPUs
- 🔄 **Auto Model Download** - Models cached in persistent volumes

## Quick Start

### Deploy to RunPod

1. Click the RunPod badge above or visit the [Hub page](https://console.runpod.io/hub/nikserg/wan-video-runpod-serverless)
2. Choose a preset based on your GPU VRAM:
   - **VACE-1.3B** - Low VRAM (3.5GB+)
   - **I2V-14B-480P** - Mid VRAM (17GB+)  
   - **I2V-14B-720P** - Consumer GPU (24GB)
   - **TI2V-5B** - Consumer GPU (24GB)
   - **I2V-A14B/T2V-A14B** - High-end (80GB+)
3. Configure LoRA URLs if needed
4. Deploy and wait for model initialization


### API Usage

```python
import requests
import base64

# Text-to-Video (720p)
response = requests.post("YOUR_ENDPOINT_URL", json={
    "input": {
        "prompt": "A beautiful sunset over the ocean with waves crashing",
        "resolution_preset": "720p",  # Auto sets to 1280x720
        "duration_seconds": 5.0,
        "fps": 24
    }
})

# Image-to-Video (480p for faster generation)
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post("YOUR_ENDPOINT_URL", json={
    "input": {
        "prompt": "Camera slowly pans across the landscape", 
        "image": image_b64,
        "resolution_preset": "480p",  # Auto sets to 854x480
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
| `resolution_preset` | string | ❌ | - | Resolution preset (480p, 720p, 720p_vertical, etc.) |
| `duration_seconds` | float | ❌ | 5.0 | Video length in seconds |
| `fps` | integer | ❌ | 24 | Frames per second (1-60) |
| `width` | integer | ❌ | 1280 | Video width in pixels (overridden by preset) |
| `height` | integer | ❌ | 720 | Video height in pixels (overridden by preset) |
| `negative_prompt` | string | ❌ | "" | What to avoid in generation |
| `guidance_scale` | float | ❌ | 5.0 | How closely to follow prompt |
| `num_inference_steps` | integer | ❌ | 50 | Quality vs speed tradeoff |
| `seed` | integer | ❌ | random | Random seed for reproducibility |

### Resolution Presets

| Preset | Resolution | Aspect Ratio | Best For |
|--------|------------|--------------|----------|
| `480p` | 854×480 | 16:9 | Fast generation, low VRAM |
| `720p` | 1280×720 | 16:9 | High quality, standard |
| `720p_vertical` | 720×1280 | 9:16 | Mobile content, stories |
| `480p_vertical` | 480×854 | 9:16 | Fast mobile content |
| `square` | 720×720 | 1:1 | Social media posts |
| `square_small` | 480×480 | 1:1 | Quick social content |

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

## WAN Model Variants

### Low VRAM Options

#### VACE-1.3B (Fastest)
- **GPU**: 3.5GB+ VRAM (GTX 1060, RTX 3060)
- **Features**: Image-to-video generation
- **Performance**: ~4 minutes for 5-second video on RTX 4090
- **Resolution**: Up to 832×480
- **Best for**: Quick prototyping, low-end hardware

### Consumer GPU Options (24GB VRAM)

#### I2V-14B-720P (Recommended for 24GB)
- **GPU**: RTX 4090, RTX 6000 Ada (24GB VRAM)
- **Features**: High-quality image-to-video
- **Performance**: Optimized for 720p generation
- **Resolution**: Up to 1280×720
- **Best for**: High-quality I2V on consumer GPUs

#### I2V-14B-480P (Memory Efficient)
- **GPU**: RTX 3090, RTX 4080 (17GB+ VRAM)
- **Features**: Image-to-video generation
- **Performance**: Faster than 720p variant
- **Resolution**: Up to 854×480
- **Best for**: Faster generation on mid-range GPUs

#### TI2V-5B (Versatile)
- **GPU**: RTX 4090+ (24GB VRAM)
- **Features**: Text-to-video + Image-to-video
- **Performance**: ~9 minutes for 720p video
- **Resolution**: Up to 1280×720
- **Best for**: Both T2V and I2V workflows

### Enterprise GPU Options (80GB+ VRAM)

#### T2V-A14B (Text-to-Video Specialist)
- **GPU**: A100, H100 (80GB+ VRAM)
- **Features**: Text-to-video only
- **Performance**: Superior quality and speed
- **Best for**: High-end T2V generation

#### I2V-A14B (Image-to-Video Specialist)
- **GPU**: A100, H100 (80GB+ VRAM)
- **Features**: Image-to-video only
- **Performance**: Superior quality and speed
- **Best for**: Professional I2V workflows

## Hardware Requirements

| Model | Min VRAM | Recommended GPU | Resolution | Performance |
|-------|----------|-----------------|------------|-------------|
| VACE-1.3B | 3.5GB | GTX 1060, RTX 3060 | 832×480 | ~4min (RTX 4090) |
| I2V-14B-480P | 17GB | RTX 3090, RTX 4080 | 854×480 | Fast generation |
| I2V-14B-720P | 24GB | RTX 4090, RTX 6000 | 1280×720 | High quality |
| TI2V-5B | 24GB | RTX 4090 | 1280×720 | ~9min |
| T2V-A14B | 80GB | A100/H100 | 1280×720+ | Superior |
| I2V-A14B | 80GB | A100/H100 | 1280×720+ | Superior |

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

### Text-to-Video (720p)
```json
{
  "input": {
    "prompt": "A cat wearing sunglasses riding a skateboard in slow motion",
    "resolution_preset": "720p",
    "duration_seconds": 4.0,
    "fps": 24
  }
}
```

### Image-to-Video (480p for speed)
```json
{
  "input": {
    "prompt": "The flowers gently sway in the wind",
    "image": "base64_encoded_flower_image",
    "resolution_preset": "480p",
    "duration_seconds": 6.0,
    "fps": 30,
    "negative_prompt": "static, still, frozen",
    "guidance_scale": 7.0,
    "seed": 12345
  }
}
```

### Vertical Video for Mobile
```json
{
  "input": {
    "prompt": "Person walking through a busy city street",
    "resolution_preset": "720p_vertical",
    "duration_seconds": 8.0,
    "fps": 24
  }
}
```

### Square Video for Social Media
```json
{
  "input": {
    "prompt": "Coffee brewing in slow motion with steam rising",
    "image": "base64_encoded_coffee_image",
    "resolution_preset": "square",
    "duration_seconds": 5.0,
    "fps": 30
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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Model Licenses:**
- WAN 2.2 models: Apache 2.0 License
- See individual model repositories on Hugging Face for specific licensing terms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with mock generator
5. Submit a pull request

## Support

- 🐛 [Report Issues](https://github.com/nikserg/wan-video-runpod-serverless/issues)
- 💬 [RunPod Community](https://discord.gg/runpod)