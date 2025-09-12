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
   - **VACE-1.3B** - Low VRAM (3.5GB+) - Fastest for testing
   - **I2V-14B-480P** - Mid VRAM (17GB+) - Good speed/quality balance  
   - **I2V-14B-720P** - Consumer GPU (24GB) - **Recommended for RTX 4090**
   - **TI2V-5B** - Consumer GPU (24GB) - Supports both T2V and I2V
   - **I2V-A14B/T2V-A14B** - High-end (80GB+) - Maximum quality
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

Configure your serverless endpoint with these environment variables:

| Variable | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| **Model Configuration** | | | | |
| `WAN_MODEL_TYPE` | select | `TI2V-5B` | Choose WAN model variant | `VACE-1.3B`, `I2V-14B-720P`, `TI2V-5B` |
| `USE_MOCK_GENERATOR` | boolean | `false` | Use mock generator for testing | `true` for testing, `false` for production |
| **LoRA Configuration** | | | | |
| `LORA_URL_1` | string | - | First LoRA download URL | `https://civitai.com/models/123?modelVersionId=456` |
| `LORA_URL_2` | string | - | Second LoRA download URL (optional) | `https://civitai.com/api/download/models/456` |
| `LORA_URL_3` | string | - | Third LoRA download URL (optional) | CivitAI model URL |
| `CIVITAI_API_KEY` | string | - | **🔐 CivitAI API key for authenticated downloads** | Get from https://civitai.com/user/account |

### Model Type Options

| Model Type | VRAM Required | Best For | Speed |
|------------|---------------|----------|-------|
| `VACE-1.3B` | 3.5GB+ | Quick testing, low-end GPUs | ⚡⚡⚡ Fastest |
| `I2V-14B-480P` | 17GB+ | Budget production, speed priority | ⚡⚡ Fast |
| `I2V-14B-720P` | 24GB+ | **Production recommended** | ⚡ Good |
| `TI2V-5B` | 24GB+ | Both T2V and I2V support | ⚡ Good |
| `T2V-A14B` | 80GB+ | Enterprise T2V only | ⚡⚡ Fast |
| `I2V-A14B` | 80GB+ | Enterprise I2V only | ⚡⚡ Fast |

## Model Capabilities

| Model | Type | Input Support | Max Resolution | Key Features |
|-------|------|---------------|----------------|--------------|
| **VACE-1.3B** | I2V Only | Image + Prompt | 832×480 | Ultra-fast, low VRAM |
| **I2V-14B-480P** | I2V Only | Image + Prompt | 854×480 | Good speed/quality balance |
| **I2V-14B-720P** | I2V Only | Image + Prompt | 1280×720 | High quality, consumer GPU |
| **TI2V-5B** | T2V + I2V | Text or Image + Prompt | 1280×720 | Versatile, dual-mode |
| **T2V-A14B** | T2V Only | Text + Prompt | 1280×720+ | Enterprise text-to-video |
| **I2V-A14B** | I2V Only | Image + Prompt | 1280×720+ | Enterprise image-to-video |

## Hardware Requirements

| Model | Min VRAM | Recommended GPU | Resolution | Performance |
|-------|----------|-----------------|------------|-------------|
| VACE-1.3B | 3.5GB | GTX 1060, RTX 3060 | 832×480 | ~1min (RTX 4090) |
| I2V-14B-480P | 17GB | RTX 3090, RTX 4080 | 854×480 | ~3-5min |
| I2V-14B-720P | 24GB | RTX 4090, RTX 6000 | 1280×720 | ~6-9min |
| TI2V-5B | 24GB | RTX 4090 | 1280×720 | ~8-12min |
| T2V-A14B | 80GB | A100/H100 | 1280×720+ | ~4-6min |
| I2V-A14B | 80GB | A100/H100 | 1280×720+ | ~4-6min |

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

## Performance Guide

### Model Selection by Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Quick Testing** | VACE-1.3B | Fastest generation (~1 min) |
| **Production I2V** | I2V-14B-720P | Best balance of speed/quality for 24GB GPU |
| **Both T2V & I2V** | TI2V-5B | Versatile, supports both modes |
| **Maximum Quality** | I2V-A14B / T2V-A14B | Enterprise-grade results |
| **Budget GPU** | I2V-14B-480P | Works on 17GB+ GPUs |

### Speed Optimization Tips

**For Faster Generation:**
- Use `480p` resolution preset instead of `720p` 
- Reduce `duration_seconds` (1-3 seconds)
- Lower `fps` (12-16 instead of 24-30)
- Reduce `num_inference_steps` to 30-40

**For Better Quality:**
- Use `720p` or higher resolution presets
- Increase `guidance_scale` to 7-10
- Use specific, detailed prompts
- Add negative prompts to avoid unwanted elements

### Cost Estimation (RunPod)

**RTX 4090 (~$0.50/hour):**
- VACE-1.3B: ~$0.01 per video
- I2V-14B-720P: ~$0.05-0.08 per video  
- TI2V-5B: ~$0.07-0.10 per video

**A100 (~$1.50/hour):**
- I2V-A14B: ~$0.10-0.15 per video
- T2V-A14B: ~$0.10-0.15 per video

## Best Practices

### Prompt Engineering
- **Be specific**: "A red sports car driving on a mountain road at sunset" vs "car driving"
- **Include motion**: "slowly rotating", "gently swaying", "smoothly moving"
- **Set the scene**: "cinematic lighting", "4K quality", "detailed background"
- **Use negative prompts**: "blurry, low quality, distorted, static"

### Optimal Settings by Content

**Nature/Landscapes:**
```json
{
  "resolution_preset": "720p",
  "fps": 16,
  "duration_seconds": 4.0,
  "guidance_scale": 6.0
}
```

**People/Portraits:**
```json
{
  "resolution_preset": "480p_vertical", 
  "fps": 20,
  "duration_seconds": 3.0,
  "guidance_scale": 7.5
}
```

**Social Media:**
```json
{
  "resolution_preset": "square",
  "fps": 24,
  "duration_seconds": 2.0,
  "guidance_scale": 8.0
}
```


### Mock Generator Mode
For testing without GPU requirements:
```bash
docker run -e USE_MOCK_GENERATOR=true wan-video-serverless
```

## Project Structure

After cloning, the project is organized as follows:

```
wan-video-serverless/
├── app/                    # Application code
│   ├── rp_handler.py      # RunPod serverless handler
│   ├── model_downloader.py # WAN 2.2 and LoRA downloader
│   ├── video_generator.py  # WAN model wrapper
│   └── check_cuda.py      # CUDA availability check
├── wan/                   # WAN 2.2 repository (cloned)
│   ├── wan/              # WAN modules
│   │   ├── configs/      # Model configurations
│   │   ├── text2video.py # WanT2V class
│   │   ├── image2video.py# WanI2V class
│   │   └── textimage2video.py # WanTI2V class
│   └── ...
├── .runpod/              # RunPod Hub configuration
│   ├── hub.json          # Hub marketplace settings
│   └── tests.json        # Test configurations
├── Dockerfile            # Container build instructions
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## LoRA Downloads & Authentication

### CivitAI API Key Setup

Many LoRAs on CivitAI require authentication. To download them automatically:

1. **Get your API key:**
   - Visit https://civitai.com/user/account
   - Create an API key
   - Copy the key

2. **Set the environment variable:**
   ```bash
   CIVITAI_API_KEY=your_api_key_here
   ```

3. **Or use RunPod Hub UI:**
   - Enter the API key in the "CivitAI API Key" field when deploying

### LoRA URL Formats

✅ **Supported URL formats:**
```bash
# Model page with version ID (recommended)
LORA_URL_1=https://civitai.com/models/1307155/model-name?modelVersionId=1475095

# Direct API download URL
LORA_URL_2=https://civitai.com/api/download/models/1475095
```

❌ **Unsupported formats:**
```bash
# Model page without version ID
LORA_URL_1=https://civitai.com/models/1307155/model-name
```

### Manual LoRA Installation

If automatic download fails, you can manually install LoRAs:

1. Download the `.safetensors` file from CivitAI
2. Place it in `/runpod-volume/loras/` directory
3. The system will automatically detect and load it

## Troubleshooting

### LoRA Download Issues

**Problem:** Getting HTML instead of LoRA file
```
❌ Downloaded HTML authentication page instead of LoRA file
```

**Solutions:**
1. **Set API Key:** Export `CIVITAI_API_KEY=your_key`
2. **Check URL format:** Use `?modelVersionId=123` parameter
3. **Manual download:** Place `.safetensors` in `/runpod-volume/loras/`
4. **Skip LoRA:** The system will continue without failed LoRAs

**Problem:** WAN import errors
```
❌ Could not import WAN modules: cannot import name 'WanT2V'
```

**Solutions:**
1. **Check project structure:** Ensure `/wan` directory exists
2. **Rebuild container:** Force rebuild with `--no-cache`
3. **Check logs:** Look for WAN 2.2 repository clone errors

**Problem:** Configuration errors
```
❌ 'EasyDict' object has no attribute 'boundary'
```

**Solutions:**
1. **Update dependencies:** Ensure `easydict` is installed
2. **Check model type:** Verify `WAN_MODEL_TYPE` is supported
3. **Use mock mode:** Set `USE_MOCK_GENERATOR=true` for testing

### Performance Issues

**Problem:** Out of memory errors
- Try smaller model (VACE-1.3B)
- Reduce resolution preset  
- Decrease `duration_seconds`
- Lower `num_inference_steps`

**Problem:** Slow generation
- Use appropriate model for your GPU
- Enable `convert_model_dtype=True`
- Set `t5_cpu=True` for TI2V-5B

## FAQ

### General Questions

**Q: Which model should I choose for production?**  
A: I2V-14B-720P for RTX 4090, or I2V-A14B for enterprise GPUs.

**Q: How long does video generation take?**  
A: 1-12 minutes depending on model and settings. See Performance table above.

**Q: Can I use custom LoRAs?**  
A: Yes! Set `LORA_URL_1`, `LORA_URL_2`, `LORA_URL_3` with CivitAI URLs. For authenticated downloads, also set `CIVITAI_API_KEY`. See LoRA Downloads section for details.

**Q: What's the maximum video length?**  
A: Technically unlimited, but longer videos require more VRAM and time. Recommended: 1-10 seconds.

**Q: Do I need to specify both width/height and resolution_preset?**  
A: No, resolution_preset overrides width/height. Use preset for convenience.

### Technical Questions

**Q: Why does the container take so long to start?**  
A: Models are downloaded on first run (20GB+). Subsequent starts are faster with volume caching.

**Q: Can I run this locally without RunPod?**  
A: Yes, but you need a GPU with sufficient VRAM. See Local Development section.

**Q: Is the API compatible with OpenAI format?**  
A: No, it uses RunPod serverless format. See API Reference for exact schema.

**Q: How do I reduce costs?**  
A: Use VACE-1.3B for testing, batch multiple requests, and optimize settings per Performance Guide.

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