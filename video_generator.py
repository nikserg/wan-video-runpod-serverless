import os
import sys
import torch
import base64
import tempfile
import logging
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# WAN 2.2 should be installed as package, no need to add to path manually

try:
    import wan
    # Import the actual modules from WAN 2.2
    from wan.text2video import Text2Video
    from wan.image2video import Image2Video
    from wan.textimage2video import TextImage2Video
    WanVideo = None  # Will be set based on model type
except ImportError as e:
    logging.warning(f"Could not import WAN modules: {e}")
    Text2Video = None
    Image2Video = None
    TextImage2Video = None
    WanVideo = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self, model_path, model_type="TI2V-5B", device="cuda", lora_paths=None):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.lora_paths = lora_paths or []
        self.model = None
        self.loaded_loras = []
        self.load_model()
    
    def get_resolution_preset(self, resolution_preset):
        """Get width and height from resolution preset"""
        resolutions = {
            "480p": (854, 480),
            "720p": (1280, 720),
            "720p_vertical": (720, 1280),
            "480p_vertical": (480, 854),
            "square": (720, 720),
            "square_small": (480, 480)
        }
        
        if resolution_preset in resolutions:
            return resolutions[resolution_preset]
        else:
            # Default to 720p if preset not found
            logger.warning(f"Unknown resolution preset: {resolution_preset}, defaulting to 720p")
            return resolutions["720p"]
    
    def validate_resolution_for_model(self, width, height, model_type):
        """Validate if resolution is supported by the model"""
        # Model-specific resolution constraints
        if model_type == "I2V-14B-480P":
            if max(width, height) > 854:
                logger.warning(f"Model {model_type} is optimized for 480p, reducing resolution from {width}x{height}")
                if width >= height:
                    return 854, 480
                else:
                    return 480, 854
        elif model_type == "VACE-1.3B":
            # VACE-1.3B supports 480x832 or 832x480
            if width >= height:
                return min(832, width), min(480, height)
            else:
                return min(480, width), min(832, height)
        
        return width, height
    
    def load_model(self):
        """Load WAN model"""
        try:
            # Select appropriate WAN class based on model type
            if Text2Video is None and Image2Video is None and TextImage2Video is None:
                raise ImportError("WAN2 modules not available")
            
            logger.info(f"Loading WAN {self.model_type} model from {self.model_path}")
            
            # Determine which WAN class to use based on model type
            if self.model_type in ["TI2V-5B"]:  # Supports both T2V and I2V
                WanClass = TextImage2Video if TextImage2Video else Text2Video
            elif self.model_type in ["T2V-A14B"]:  # Text-to-Video only
                WanClass = Text2Video
            elif self.model_type in ["I2V-A14B", "I2V-14B-720P", "I2V-14B-480P", "VACE-1.3B"]:  # Image-to-Video only
                WanClass = Image2Video
            else:
                # Default to TextImage2Video for unknown models
                WanClass = TextImage2Video if TextImage2Video else Text2Video
            
            if WanClass is None:
                raise ImportError(f"No suitable WAN class found for model type: {self.model_type}")
            
            # Configure model parameters based on type
            if self.model_type in ["TI2V-5B"]:
                # WAN 2.2 TI2V-5B - Consumer GPU friendly
                config = {
                    "ckpt_dir": self.model_path,
                    "offload_model": True,
                    "convert_model_dtype": True,
                    "t5_cpu": True,
                }
            elif self.model_type in ["I2V-14B-720P", "I2V-14B-480P"]:
                # WAN 2.1 I2V models - 24GB VRAM optimized
                config = {
                    "ckpt_dir": self.model_path,
                    "offload_model": True,
                    "convert_model_dtype": True,
                    "enable_vae_slicing": True,
                }
            elif self.model_type == "VACE-1.3B":
                # WAN 2.1 VACE - Very efficient model
                config = {
                    "ckpt_dir": self.model_path,
                    "offload_model": False,  # Small enough to keep in VRAM
                    "convert_model_dtype": True,
                }
            else:  # A14B and other high-end models
                config = {
                    "ckpt_dir": self.model_path,
                    "offload_model": True,
                    "convert_model_dtype": True,
                }
            
            self.model = WanClass(**config)
            logger.info("Model loaded successfully")
            
            # Load LoRAs if provided
            self.load_loras()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_loras(self):
        """Load LoRAs with memory optimization"""
        if not self.lora_paths or self.model is None:
            return
        
        try:
            for lora_path in self.lora_paths:
                if not os.path.exists(lora_path):
                    logger.warning(f"LoRA file not found: {lora_path}")
                    continue
                
                logger.info(f"Loading LoRA: {lora_path}")
                
                # Load LoRA with diffusers if available
                if hasattr(self.model, 'load_lora_weights'):
                    # For diffusers-based models
                    self.model.load_lora_weights(lora_path)
                    self.loaded_loras.append(lora_path)
                    logger.info(f"LoRA loaded successfully: {lora_path}")
                elif hasattr(self.model, 'load_adapter'):
                    # Alternative LoRA loading method
                    self.model.load_adapter(lora_path)
                    self.loaded_loras.append(lora_path)
                    logger.info(f"LoRA adapter loaded: {lora_path}")
                else:
                    logger.warning(f"Model does not support LoRA loading: {type(self.model)}")
                    
        except Exception as e:
            logger.error(f"Failed to load LoRAs: {e}")
    
    def unload_loras(self):
        """Unload LoRAs to free memory"""
        if not self.loaded_loras or self.model is None:
            return
        
        try:
            if hasattr(self.model, 'unload_lora_weights'):
                self.model.unload_lora_weights()
                logger.info(f"Unloaded {len(self.loaded_loras)} LoRAs")
            elif hasattr(self.model, 'unload_adapter'):
                for lora_path in self.loaded_loras:
                    self.model.unload_adapter()
                logger.info(f"Unloaded {len(self.loaded_loras)} LoRA adapters")
            
            self.loaded_loras = []
            
            # Force garbage collection to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Failed to unload LoRAs: {e}")
    
    def preprocess_image(self, image_data):
        """Preprocess image for WAN 2.2"""
        if isinstance(image_data, str):
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data)
            import io
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, bytes):
            import io
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def calculate_frames(self, duration_seconds, fps):
        """Calculate number of frames from duration and FPS"""
        return int(duration_seconds * fps)
    
    def generate_video(self, 
                      prompt,
                      image=None,
                      negative_prompt="",
                      width=1280,
                      height=720,
                      duration_seconds=5.0,
                      fps=24,
                      guidance_scale=5.0,
                      num_inference_steps=50,
                      seed=None,
                      resolution_preset=None):
        """Generate video using WAN models"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Handle resolution preset if provided
            if resolution_preset:
                width, height = self.get_resolution_preset(resolution_preset)
                logger.info(f"Using resolution preset '{resolution_preset}': {width}x{height}")
            
            # Validate resolution for model
            width, height = self.validate_resolution_for_model(width, height, self.model_type)
            
            # Calculate number of frames from duration and FPS
            num_frames = self.calculate_frames(duration_seconds, fps)
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Determine generation type based on model capabilities
            i2v_models = ["TI2V-5B", "I2V-A14B", "I2V-14B-720P", "I2V-14B-480P", "VACE-1.3B"]
            t2v_models = ["T2V-A14B", "TI2V-5B"]
            
            if image is not None and self.model_type in i2v_models:
                # Image-to-video generation
                processed_image = self.preprocess_image(image)
                
                logger.info(f"Generating I2V with {self.model_type}: {width}x{height}, {duration_seconds}s ({num_frames} frames), {fps} FPS")
                
                video_frames = self.model.generate(
                    input_prompt=prompt,
                    img=processed_image,
                    max_area=width*height,
                    frame_num=num_frames,
                    guide_scale=guidance_scale,
                    sampling_steps=num_inference_steps,
                    n_prompt=negative_prompt,
                    seed=seed if seed is not None else None
                )
            elif image is None and self.model_type in t2v_models:
                # Text-to-video generation
                logger.info(f"Generating T2V with {self.model_type}: {width}x{height}, {duration_seconds}s ({num_frames} frames), {fps} FPS")
                
                video_frames = self.model.generate(
                    input_prompt=prompt,
                    size=(width, height),
                    frame_num=num_frames,
                    guide_scale=guidance_scale,
                    sampling_steps=num_inference_steps,
                    n_prompt=negative_prompt,
                    seed=seed if seed is not None else None
                )
            else:
                # Model doesn't support this mode
                if image is not None:
                    raise ValueError(f"Model {self.model_type} does not support image-to-video generation")
                else:
                    raise ValueError(f"Model {self.model_type} does not support text-to-video generation")
            
            # Convert frames to video
            return self.frames_to_mp4(video_frames, fps)
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
    
    def frames_to_mp4(self, frames, fps=24):
        """Convert frames to MP4 and return as base64"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Convert frames to numpy arrays if needed
            if torch.is_tensor(frames):
                frames = frames.cpu().numpy()
            
            # WAN returns frames in (C, N, H, W) format, convert to (N, H, W, C)
            if frames.ndim == 4:
                if frames.shape[0] in [1, 3]:  # (C, N, H, W) format
                    frames = frames.transpose(1, 2, 3, 0)  # (N, H, W, C)
                # else assume already in (N, H, W, C) format
            
            # Normalize to [0, 1] if needed, then convert to uint8
            if frames.dtype == np.float32 or frames.dtype == np.float64:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = frames.astype(np.uint8)
            elif frames.dtype != np.uint8:
                frames = frames.astype(np.uint8)
            
            # Get video dimensions
            num_frames, height, width = frames.shape[:3]
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                if frame.shape[-1] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            
            # Read the video file and encode to base64
            with open(tmp_path, 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            logger.info(f"Generated MP4 video: {len(video_bytes)} bytes")
            return video_base64
            
        except Exception as e:
            logger.error(f"Failed to convert frames to MP4: {e}")
            raise

# Fallback implementation for testing without actual WAN2 model
class MockVideoGenerator:
    def __init__(self, model_path, model_type="TI2V-5B", device="cuda"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        logger.info("Using mock video generator for testing")
    
    def calculate_frames(self, duration_seconds, fps):
        """Calculate number of frames from duration and FPS"""
        return int(duration_seconds * fps)
    
    def generate_video(self, prompt, image=None, duration_seconds=5.0, fps=24, resolution_preset=None, **kwargs):
        """Generate a mock video for testing"""
        try:
            # Handle resolution preset
            width = kwargs.get('width', 1280)
            height = kwargs.get('height', 720)
            
            if resolution_preset:
                resolutions = {
                    "480p": (854, 480),
                    "720p": (1280, 720),
                    "720p_vertical": (720, 1280),
                    "480p_vertical": (480, 854),
                    "square": (720, 720),
                    "square_small": (480, 480)
                }
                if resolution_preset in resolutions:
                    width, height = resolutions[resolution_preset]
                    logger.info(f"Mock generator using resolution preset '{resolution_preset}': {width}x{height}")
            
            # Calculate frames from duration
            num_frames = self.calculate_frames(duration_seconds, fps)
            
            logger.info(f"Generating mock video: {width}x{height}, {duration_seconds}s ({num_frames} frames), {fps} FPS")
            
            # Create colored frames
            frames = []
            for i in range(min(num_frames, 120)):  # Limit to 120 frames for testing
                # Create a colored frame with frame number
                color = (i * 8 % 255, (i * 16) % 255, (i * 32) % 255)
                frame = np.full((height, width, 3), color, dtype=np.uint8)
                
                # Add text overlay with resolution info
                cv2.putText(frame, f"Frame {i+1}/{num_frames}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(frame, f"{width}x{height} @ {fps}fps", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if resolution_preset:
                    cv2.putText(frame, f"Preset: {resolution_preset}", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, prompt[:40], (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, prompt[:50], (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frames.append(frame)
            
            return self.frames_to_mp4(frames, fps)
            
        except Exception as e:
            logger.error(f"Mock video generation failed: {e}")
            raise
    
    def frames_to_mp4(self, frames, fps=24):
        """Convert frames to MP4 and return as base64"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            with open(tmp_path, 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            os.unlink(tmp_path)
            
            logger.info(f"Generated mock MP4 video: {len(video_bytes)} bytes")
            return video_base64
            
        except Exception as e:
            logger.error(f"Failed to convert mock frames to MP4: {e}")
            raise

def create_generator(model_path, model_type="TI2V-5B", device="cuda", use_mock=False, lora_paths=None):
    """Factory function to create video generator"""
    wan_available = Text2Video is not None or Image2Video is not None or TextImage2Video is not None
    if use_mock or not wan_available:
        return MockVideoGenerator(model_path, model_type, device)
    else:
        return VideoGenerator(model_path, model_type, device, lora_paths)