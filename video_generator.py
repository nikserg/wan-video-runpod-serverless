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

# Add WAN 2.2 to path
sys.path.append("/workspace/Wan2.2")

try:
    from wan2 import WanVideo
    from wan2.utils import save_video
except ImportError as e:
    logging.warning(f"Could not import WAN2 modules: {e}")
    WanVideo = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self, model_path, model_type="TI2V-5B", device="cuda"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load WAN 2.2 model"""
        try:
            if WanVideo is None:
                raise ImportError("WAN2 modules not available")
            
            logger.info(f"Loading WAN 2.2 {self.model_type} model from {self.model_path}")
            
            # Configure model parameters based on type
            if self.model_type == "TI2V-5B":
                # Consumer GPU friendly settings
                config = {
                    "ckpt_dir": self.model_path,
                    "offload_model": True,
                    "convert_model_dtype": True,
                    "t5_cpu": True,
                }
            else:  # A14B models
                config = {
                    "ckpt_dir": self.model_path,
                    "offload_model": True,
                    "convert_model_dtype": True,
                }
            
            self.model = WanVideo(**config)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_data):
        """Preprocess image for WAN 2.2"""
        if isinstance(image_data, str):
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(tempfile.BytesIO(image_bytes))
        elif isinstance(image_data, bytes):
            image = Image.open(tempfile.BytesIO(image_data))
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
                      seed=None):
        """Generate video using WAN 2.2"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Calculate number of frames from duration and FPS
            num_frames = self.calculate_frames(duration_seconds, fps)
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Determine generation type
            if image is not None and self.model_type in ["TI2V-5B", "I2V-A14B"]:
                # Image-to-video or text-image-to-video
                processed_image = self.preprocess_image(image)
                
                logger.info(f"Generating I2V/TI2V: {width}x{height}, {duration_seconds}s ({num_frames} frames), {fps} FPS")
                
                video_frames = self.model.generate(
                    prompt=prompt,
                    image=processed_image,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            else:
                # Text-to-video
                logger.info(f"Generating T2V: {width}x{height}, {duration_seconds}s ({num_frames} frames), {fps} FPS")
                
                video_frames = self.model.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            
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
            
            # Ensure frames are in correct format (H, W, C) and uint8
            if frames.ndim == 4:  # (T, H, W, C)
                frames = frames.transpose(0, 1, 2, 3)
            
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8)
            
            # Get video dimensions
            height, width = frames.shape[1:3]
            
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
    
    def generate_video(self, prompt, image=None, duration_seconds=5.0, fps=24, **kwargs):
        """Generate a mock video for testing"""
        try:
            # Calculate frames from duration
            num_frames = self.calculate_frames(duration_seconds, fps)
            width = kwargs.get('width', 1280)
            height = kwargs.get('height', 720)
            
            logger.info(f"Generating mock video: {width}x{height}, {duration_seconds}s ({num_frames} frames), {fps} FPS")
            
            # Create colored frames
            frames = []
            for i in range(min(num_frames, 120)):  # Limit to 120 frames for testing
                # Create a colored frame with frame number
                color = (i * 8 % 255, (i * 16) % 255, (i * 32) % 255)
                frame = np.full((height, width, 3), color, dtype=np.uint8)
                
                # Add text overlay
                cv2.putText(frame, f"Frame {i+1}/{num_frames}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(frame, f"{duration_seconds}s @ {fps}fps", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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

def create_generator(model_path, model_type="TI2V-5B", device="cuda", use_mock=False):
    """Factory function to create video generator"""
    if use_mock or WanVideo is None:
        return MockVideoGenerator(model_path, model_type, device)
    else:
        return VideoGenerator(model_path, model_type, device)