import runpod
import base64
import logging
import os
from model_downloader import ModelDownloader
from video_generator import create_generator
from check_cuda import is_cuda_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
generator = None
model_info = None

if not is_cuda_available():
    logger.error("CUDA is not available.")
    exit(1)


def initialize_models():
    """Initialize WAN 2.2 models and LoRAs on container startup"""
    global generator, model_info
    
    try:
        logger.info("Initializing WAN 2.2 models...")
        
        # Download models and LoRAs
        downloader = ModelDownloader()
        model_info = downloader.setup_models()
        
        logger.info(f"Model setup complete: {model_info}")
        
        # Initialize video generator
        use_mock = os.getenv("USE_MOCK_GENERATOR", "false").lower() == "true"
        generator = create_generator(
            model_path=model_info["model_path"],
            model_type=model_info["model_type"],
            use_mock=use_mock,
            lora_paths=model_info.get("lora_paths", [])
        )
        
        logger.info("Video generator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Use mock generator as fallback
        generator = create_generator("", "TI2V-5B", use_mock=True)

def handler(job):
    """RunPod serverless handler for WAN 2.2 video generation"""
    global generator
    
    if generator is None:
        initialize_models()
    
    try:
        input_data = job["input"]
        
        # Required parameters
        prompt = input_data.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is required"}
        
        # Optional parameters with defaults
        image_base64 = input_data.get("image")
        negative_prompt = input_data.get("negative_prompt", "")
        resolution_preset = input_data.get("resolution_preset")  # e.g., "480p", "720p"
        width = input_data.get("width", 1280)
        height = input_data.get("height", 720)
        duration_seconds = input_data.get("duration_seconds", 5.0)  # Video length in seconds
        fps = input_data.get("fps", 24)
        guidance_scale = input_data.get("guidance_scale", 5.0)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        seed = input_data.get("seed")
        
        # Validate parameters
        if duration_seconds <= 0:
            return {"error": "duration_seconds must be greater than 0"}
        
        if fps <= 0 or fps > 60:
            return {"error": "fps must be between 1 and 60"}
        
        if width <= 0 or height <= 0:
            return {"error": "width and height must be greater than 0"}
        
        # Validate resolution preset if provided
        valid_presets = ["480p", "720p", "720p_vertical", "480p_vertical", "square", "square_small"]
        if resolution_preset and resolution_preset not in valid_presets:
            return {"error": f"Invalid resolution_preset. Valid options: {valid_presets}"}
        
        # Process image if provided
        image_data = None
        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
            except Exception as e:
                return {"error": f"Invalid image base64: {e}"}
        
        logger.info(f"Generating video: {prompt[:50]}... ({width}x{height}, {duration_seconds}s @ {fps}fps)")
        
        # Generate video
        video_base64 = generator.generate_video(
            prompt=prompt,
            image=image_data,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            resolution_preset=resolution_preset
        )
        
        return {
            "video_base64": video_base64,
            "prompt": prompt,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "has_input_image": image_data is not None,
            "model_type": model_info.get("model_type", "unknown") if model_info else "mock",
            "loras_loaded": len(model_info.get("lora_paths", [])) if model_info else 0
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

# Initialize models when the container starts
initialize_models()

runpod.serverless.start({"handler": handler})
