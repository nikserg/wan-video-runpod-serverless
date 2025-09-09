import os
import sys
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse
from huggingface_hub import snapshot_download
import zipfile
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, models_dir="/models", loras_dir="/loras"):
        self.models_dir = Path(models_dir)
        self.loras_dir = Path(loras_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loras_dir.mkdir(parents=True, exist_ok=True)
    
    def download_wan_model(self, model_name="Wan-AI/Wan2.2-TI2V-5B", force_download=False):
        """Download WAN 2.2 model from Hugging Face"""
        model_path = self.models_dir / model_name.split("/")[-1]
        
        if model_path.exists() and not force_download:
            logger.info(f"Model {model_name} already exists at {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading WAN model: {model_name}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"Successfully downloaded {model_name} to {model_path}")
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise
    
    def download_civitai_lora(self, url, filename=None):
        """Download LoRA from CivitAI URL"""
        try:
            # Extract model ID from URL
            if "civitai.com" in url:
                # Handle different CivitAI URL formats
                if "/models/" in url:
                    model_id = url.split("/models/")[1].split("/")[0]
                elif "/api/download/models/" in url:
                    model_id = url.split("/api/download/models/")[1].split("?")[0]
                else:
                    model_id = url.split("/")[-1].split("?")[0]
                
                if not filename:
                    filename = f"lora_{model_id}.safetensors"
                
                lora_path = self.loras_dir / filename
                
                if lora_path.exists():
                    logger.info(f"LoRA already exists at {lora_path}")
                    return str(lora_path)
                
                logger.info(f"Downloading LoRA from: {url}")
                
                # Download with proper headers
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"Successfully downloaded LoRA to {lora_path}")
                return str(lora_path)
            else:
                raise ValueError(f"Unsupported LoRA URL format: {url}")
                
        except Exception as e:
            logger.error(f"Failed to download LoRA from {url}: {e}")
            raise
    
    def download_loras_from_env(self):
        """Download LoRAs specified in environment variables"""
        lora_urls = []
        lora_paths = []
        
        # Check for environment variables with LoRA URLs
        for key, value in os.environ.items():
            if key.startswith("LORA_URL"):
                lora_urls.append(value)
            elif key == "LORA_URLS":
                # Support comma-separated URLs
                lora_urls.extend([url.strip() for url in value.split(",")])
        
        for i, url in enumerate(lora_urls):
            if url:
                try:
                    filename = f"lora_{i+1}.safetensors"
                    lora_path = self.download_civitai_lora(url, filename)
                    lora_paths.append(lora_path)
                except Exception as e:
                    logger.warning(f"Failed to download LoRA {i+1} from {url}: {e}")
        
        return lora_paths
    
    def setup_models(self):
        """Setup all required models and LoRAs"""
        logger.info("Setting up WAN 2.2 models and LoRAs...")
        
        # Get model type from environment variable
        model_type = os.getenv("WAN_MODEL_TYPE", "TI2V-5B")
        
        if model_type == "TI2V-5B":
            model_name = "Wan-AI/Wan2.2-TI2V-5B"
        elif model_type == "T2V-A14B":
            model_name = "Wan-AI/Wan2.2-T2V-A14B"
        elif model_type == "I2V-A14B":
            model_name = "Wan-AI/Wan2.2-I2V-A14B"
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to TI2V-5B")
            model_name = "Wan-AI/Wan2.2-TI2V-5B"
        
        # Download main model
        model_path = self.download_wan_model(model_name)
        
        # Download LoRAs from environment variables
        lora_paths = self.download_loras_from_env()
        
        return {
            "model_path": model_path,
            "lora_paths": lora_paths,
            "model_type": model_type
        }

if __name__ == "__main__":
    downloader = ModelDownloader()
    setup_info = downloader.setup_models()
    print(f"Setup completed: {setup_info}")