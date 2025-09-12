import os
import sys
import torch
import base64
import tempfile
import logging
from PIL import Image
import cv2
import numpy as np
import psutil
import gc

# Add WAN path to sys.path
wan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wan')
if wan_path not in sys.path:
    sys.path.insert(0, wan_path)

try:
    # Import the actual classes from WAN 2.2
    from wan.text2video import WanT2V
    from wan.image2video import WanI2V
    from wan.textimage2video import WanTI2V
except ImportError as e:
    logging.warning(f"Could not import WAN modules: {e}")
    WanT2V = None
    WanI2V = None
    WanTI2V = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage(stage="", device_id=0):
    """Log both GPU and RAM memory usage"""
    try:
        # GPU Memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
            gpu_memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
            logger.info(f"🔥 GPU Memory {stage}:")
            logger.info(f"   📊 Allocated: {gpu_memory_allocated:.2f}GB / {gpu_memory_total:.2f}GB ({gpu_memory_allocated/gpu_memory_total*100:.1f}%)")
            logger.info(f"   📦 Reserved: {gpu_memory_reserved:.2f}GB ({gpu_memory_reserved/gpu_memory_total*100:.1f}%)")
        else:
            logger.warning("🚫 CUDA not available")
        
        # RAM Memory
        ram_info = psutil.virtual_memory()
        ram_used = ram_info.used / 1024**3  # GB
        ram_total = ram_info.total / 1024**3  # GB
        ram_percent = ram_info.percent
        logger.info(f"💾 RAM Memory {stage}:")
        logger.info(f"   📊 Used: {ram_used:.2f}GB / {ram_total:.2f}GB ({ram_percent:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to log memory usage: {e}")

def log_model_device_location(model, model_name="Model"):
    """Log which device the model components are loaded on"""
    try:
        logger.info(f"📍 {model_name} Device Location:")
        
        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            if hasattr(model.text_encoder, 'device'):
                logger.info(f"   📝 Text Encoder: {model.text_encoder.device}")
            elif hasattr(model.text_encoder, 'model') and hasattr(model.text_encoder.model, 'device'):
                logger.info(f"   📝 Text Encoder: {model.text_encoder.model.device}")
            else:
                # Check parameters device
                try:
                    first_param = next(model.text_encoder.parameters())
                    logger.info(f"   📝 Text Encoder: {first_param.device}")
                except:
                    logger.info("   📝 Text Encoder: device unknown")
        
        if hasattr(model, 'vae') and model.vae is not None:
            try:
                first_param = next(model.vae.parameters())
                logger.info(f"   🎨 VAE: {first_param.device}")
            except:
                logger.info("   🎨 VAE: device unknown")
        
        if hasattr(model, 'model') and model.model is not None:
            try:
                first_param = next(model.model.parameters())
                logger.info(f"   🧠 Main Model: {first_param.device}")
            except:
                logger.info("   🧠 Main Model: device unknown")
        
        if hasattr(model, 'device'):
            logger.info(f"   🎯 Model Device Attribute: {model.device}")
            
    except Exception as e:
        logger.error(f"Failed to log model device location: {e}")

def force_model_to_gpu(model, target_device='cuda:0'):
    """Forcefully move all model components to GPU"""
    try:
        logger.info(f"🚀 Force moving entire model to {target_device}...")
        
        # Move the entire model if it has .to() method
        if hasattr(model, 'to'):
            model.to(target_device)
            logger.info("✅ Model moved via .to() method")
        
        # Force move all submodules
        if hasattr(model, 'modules'):
            for name, module in model.named_modules():
                if hasattr(module, 'to'):
                    module.to(target_device)
                    logger.debug(f"  ↳ {name} moved to {target_device}")
        
        # Force move all parameters
        total_params = 0
        moved_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if not param.is_cuda:
                param.data = param.data.to(target_device)
                moved_params += 1
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(target_device)
        
        logger.info(f"📊 Moved {moved_params}/{total_params} parameters to GPU")
        
        # Force move all buffers  
        total_buffers = 0
        moved_buffers = 0
        for name, buffer in model.named_buffers():
            total_buffers += 1
            if not buffer.is_cuda:
                buffer.data = buffer.data.to(target_device)
                moved_buffers += 1
                
        logger.info(f"📦 Moved {moved_buffers}/{total_buffers} buffers to GPU")
        
        # Clear CPU memory after moving
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to force model to GPU: {e}")
        return False

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
            # Log memory before model loading
            logger.info(f"🚀 Starting to load WAN {self.model_type} model from {self.model_path}")
            log_memory_usage("BEFORE model loading", device_id=0)
            
            # Select appropriate WAN class based on model type
            if WanT2V is None and WanI2V is None and WanTI2V is None:
                raise ImportError("WAN2 modules not available")
            
            logger.info(f"📦 Loading WAN {self.model_type} model from {self.model_path}")
            
            # Log target device
            logger.info(f"🎯 Target device: {self.device}")
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available. GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("❌ CUDA not available!")
            
            # Determine which WAN class to use based on model type
            if self.model_type in ["TI2V-5B"]:  # Supports both T2V and I2V
                WanClass = WanTI2V if WanTI2V else WanT2V
            elif self.model_type in ["T2V-A14B"]:  # Text-to-Video only
                WanClass = WanT2V
            elif self.model_type in ["I2V-A14B", "I2V-14B-720P", "I2V-14B-480P", "VACE-1.3B"]:  # Image-to-Video only
                WanClass = WanI2V
            else:
                # Default to TextImage2Video for unknown models
                WanClass = WanTI2V if WanTI2V else WanT2V
            
            if WanClass is None:
                raise ImportError(f"No suitable WAN class found for model type: {self.model_type}")
            
            logger.info(f"📋 Using WAN class: {WanClass.__name__}")
            
            # Configure model parameters based on type - WAN 2.2 uses different initialization
            # Need to load config from model directory and pass checkpoint_dir
            if self.model_type in ["TI2V-5B"]:
                # WAN 2.2 TI2V-5B - Consumer GPU friendly
                # Import the specific config for TI2V-5B
                try:
                    from wan.configs.wan_ti2v_5B import ti2v_5B
                    config = ti2v_5B
                    # Add any missing parameters that might be needed
                    if not hasattr(config, 'boundary'):
                        config.boundary = 0.85  # Default boundary for TI2V
                except ImportError:
                    # Fallback: create config manually
                    from easydict import EasyDict
                    from wan.configs.shared_config import wan_shared_cfg
                    config = EasyDict()
                    config.update(wan_shared_cfg)
                    # Add TI2V-5B specific parameters
                    config.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                    config.t5_tokenizer = 'google/umt5-xxl'
                    config.vae_checkpoint = 'Wan2.2_VAE.pth'
                    config.vae_stride = (4, 16, 16)
                    config.patch_size = (1, 2, 2)
                    config.dim = 3072
                    config.ffn_dim = 14336
                    config.freq_dim = 256
                    config.num_heads = 24
                    config.num_layers = 30
                    config.window_size = (-1, -1)
                    config.qk_norm = True
                    config.cross_attn_norm = True
                    config.eps = 1e-6
                    config.sample_fps = 24
                    config.sample_shift = 5.0
                    config.sample_steps = 50
                    config.sample_guide_scale = 5.0
                    config.frame_num = 121
                    config.boundary = 0.85  # Default boundary for TI2V
                model_kwargs = {
                    "config": config,
                    "checkpoint_dir": self.model_path,
                    "device_id": 0,  # Use GPU 0
                    "t5_cpu": False,  # Load T5 on GPU for faster inference
                    "init_on_cpu": False,  # Initialize directly on GPU
                    "convert_model_dtype": True,
                    "rank": 0,  # Single GPU setup
                    "t5_fsdp": False,  # Disable T5 FSDP to control device placement
                    "dit_fsdp": False,  # Disable DiT FSDP to control device placement
                }
                logger.info(f"⚙️  TI2V-5B Model Parameters:")
                logger.info(f"   🎯 device_id: {model_kwargs['device_id']}")
                logger.info(f"   🧠 t5_cpu: {model_kwargs['t5_cpu']}")
                logger.info(f"   💾 init_on_cpu: {model_kwargs['init_on_cpu']}")
                logger.info(f"   🔄 convert_model_dtype: {model_kwargs['convert_model_dtype']}")
            elif self.model_type in ["I2V-14B-720P", "I2V-14B-480P"]:
                # WAN 2.1 I2V models - 24GB VRAM optimized
                from easydict import EasyDict
                from wan.configs.shared_config import wan_shared_cfg
                config = EasyDict()
                config.update(wan_shared_cfg)
                # Add required parameters for WAN 2.1 models
                config.boundary = 0.85  # Default boundary
                config.sample_shift = 5.0
                model_kwargs = {
                    "config": config,
                    "checkpoint_dir": self.model_path,
                    "device_id": 0,  # Use GPU 0
                    "t5_cpu": False,  # Load T5 on GPU
                    "init_on_cpu": False,  # Initialize directly on GPU
                    "convert_model_dtype": True,
                }
            elif self.model_type == "VACE-1.3B":
                # WAN 2.1 VACE - Very efficient model (optimized for low memory)
                from easydict import EasyDict
                from wan.configs.shared_config import wan_shared_cfg
                config = EasyDict()
                config.update(wan_shared_cfg)
                config.frame_num = 61  # Shorter videos for efficiency
                # Add required parameters for VACE model
                config.boundary = 0.80  # Lower boundary for efficient model
                config.sample_shift = 3.0
                # VACE model specific config - ensure T5 checkpoint is available
                config.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                config.t5_tokenizer = 'google/umt5-xxl'
                
                # Check available memory and optimize VACE configuration
                ram_info = psutil.virtual_memory()
                available_ram_gb = ram_info.available / 1024**3
                
                if available_ram_gb < 12.0:  # Less than 12GB available RAM
                    logger.info(f"🔧 VACE Memory Optimization: Available RAM {available_ram_gb:.2f}GB")
                    # For VACE-1.3B, prioritize GPU to avoid RAM overload
                    model_kwargs = {
                        "config": config,
                        "checkpoint_dir": self.model_path,
                        "convert_model_dtype": True,
                        "device_id": 0,  # Use GPU 0
                        "t5_cpu": False,  # Force T5 to GPU to save RAM
                        "init_on_cpu": False,  # Direct GPU init to save RAM
                        "rank": 0,  # Single GPU
                        "dit_fsdp": False,  # Disable FSDP for simplicity
                        "t5_fsdp": False,   # Disable T5 FSDP
                    }
                else:
                    model_kwargs = {
                        "config": config,
                        "checkpoint_dir": self.model_path,
                        "convert_model_dtype": True,
                        "device_id": 0,  # Use GPU 0
                        "t5_cpu": False,  # Load T5 on GPU
                        "init_on_cpu": False,  # Initialize directly on GPU
                    }
                
                logger.info(f"⚙️  VACE-1.3B Model Parameters:")
                logger.info(f"   🎯 device_id: {model_kwargs['device_id']}")
                logger.info(f"   🧠 t5_cpu: {model_kwargs['t5_cpu']}")
                logger.info(f"   💾 init_on_cpu: {model_kwargs['init_on_cpu']}")
                logger.info(f"   🔄 convert_model_dtype: {model_kwargs['convert_model_dtype']}")
                logger.info(f"   💡 Memory optimization for {available_ram_gb:.1f}GB available RAM")
            else:  # Default fallback for other models
                from easydict import EasyDict
                from wan.configs.shared_config import wan_shared_cfg
                config = EasyDict()
                config.update(wan_shared_cfg)
                logger.warning(f"Using default config for unknown model type: {self.model_type}")
                # Add default required parameters
                config.boundary = 0.85
                config.sample_shift = 5.0
                
                # Add specific configs for A14B models if matched
                if self.model_type == "T2V-A14B":
                    try:
                        from wan.configs.wan_t2v_A14B import t2v_A14B
                        config = t2v_A14B
                    except ImportError:
                        pass  # Use shared config
                elif self.model_type == "I2V-A14B":
                    try:
                        from wan.configs.wan_i2v_A14B import i2v_A14B
                        config = i2v_A14B
                    except ImportError:
                        pass  # Use shared config
                model_kwargs = {
                    "config": config,
                    "checkpoint_dir": self.model_path,
                    "device_id": 0,  # Use GPU 0
                    "t5_cpu": False,  # Load T5 on GPU
                    "init_on_cpu": False,  # Initialize directly on GPU
                    "convert_model_dtype": True,
                }
            
            # Log memory before model instantiation
            log_memory_usage("BEFORE model instantiation", device_id=0)
            
            # Clear cache before model loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Check available RAM before model instantiation
            ram_info = psutil.virtual_memory()
            available_ram_gb = ram_info.available / 1024**3
            logger.info(f"🔍 Available RAM before loading: {available_ram_gb:.2f}GB")
            
            if available_ram_gb < 8.0:  # Less than 8GB available
                logger.warning(f"⚠️  Low available RAM: {available_ram_gb:.2f}GB. This may cause OOM!")
                
                # Force more aggressive GPU usage for low RAM systems
                if "t5_cpu" in model_kwargs:
                    model_kwargs["t5_cpu"] = False
                    logger.info("🔧 Forcing T5 to GPU due to low RAM")
                
                if "init_on_cpu" in model_kwargs:
                    model_kwargs["init_on_cpu"] = False
                    logger.info("🔧 Forcing direct GPU initialization due to low RAM")
            
            logger.info("🏗️  Instantiating model...")
            
            # Force PyTorch to load directly on GPU to avoid RAM staging
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(0)  # Set GPU 0 as default
            
            # Set environment variables to force GPU loading
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            # Force PyTorch to prefer GPU memory allocation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # Prevent CPU fallback for CUDA operations
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            # Use context manager to force GPU loading
            try:
                with torch.cuda.device(0):
                    # Set tensor creation to default to GPU
                    torch.set_default_device('cuda:0')
                    logger.info("🎯 Set default tensor device to cuda:0")
                    
                    try:
                        self.model = WanClass(**model_kwargs)
                        logger.info("✅ Model instantiated with GPU as default device")
                        
                        # Force move all model components to GPU immediately after creation
                        force_model_to_gpu(self.model, 'cuda:0')
                        
                    finally:
                        # Reset default device
                        torch.set_default_device('cpu')
                        torch.cuda.set_device(original_device)
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    logger.error("💥 GPU Out of Memory! Trying fallback configuration...")
                    # Fallback: try with T5 on CPU to save GPU memory
                    model_kwargs["t5_cpu"] = True
                    model_kwargs["init_on_cpu"] = True
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("🔄 Retrying with T5 on CPU...")
                    self.model = WanClass(**model_kwargs)
                else:
                    raise
            except Exception as e:
                logger.error(f"💥 Model instantiation failed: {e}")
                log_memory_usage("DURING ERROR", device_id=0)
                raise
            
            # Log memory after model instantiation
            log_memory_usage("AFTER model instantiation", device_id=0)
            
            # Log device locations of model components
            log_model_device_location(self.model, f"WAN {self.model_type}")
            
            logger.info("✅ Model loaded successfully")
            
            # Load LoRAs if provided
            if self.lora_paths:
                logger.info(f"🎨 Loading {len(self.lora_paths)} LoRA(s)...")
                log_memory_usage("BEFORE LoRA loading", device_id=0)
            
            self.load_loras()
            
            if self.loaded_loras:
                log_memory_usage("AFTER LoRA loading", device_id=0)
                logger.info(f"✅ Loaded {len(self.loaded_loras)} LoRA(s) successfully")
            
            # Final memory state logging
            log_memory_usage("FINAL state", device_id=0)
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(f"📊 Error type: {type(e).__name__}")
            log_memory_usage("AFTER ERROR", device_id=0)
            
            # Additional error context
            if "out of memory" in str(e).lower():
                logger.error("💡 Suggestion: Try using a smaller model (VACE-1.3B) or increase system RAM")
            elif "cuda" in str(e).lower():
                logger.error("💡 Suggestion: Check GPU availability and CUDA installation")
            elif "checkpoint" in str(e).lower():
                logger.error("💡 Suggestion: Verify model files are properly downloaded")
            
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
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
    wan_available = WanT2V is not None or WanI2V is not None or WanTI2V is not None
    if use_mock or not wan_available:
        return MockVideoGenerator(model_path, model_type, device)
    else:
        return VideoGenerator(model_path, model_type, device, lora_paths)