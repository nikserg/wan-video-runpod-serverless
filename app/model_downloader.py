import os
import logging
import requests
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, models_dir="/runpod-volume/models", loras_dir="/runpod-volume/loras"):
        self.models_dir = Path(models_dir)
        self.loras_dir = Path(loras_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loras_dir.mkdir(parents=True, exist_ok=True)
    
    def download_wan_model(self, model_name="Wan-AI/Wan2.2-TI2V-5B", force_download=False):
        """Download WAN model from Hugging Face using huggingface-cli"""
        model_path = self.models_dir / model_name.split("/")[-1]
        
        if model_path.exists() and not force_download:
            logger.info(f"Model {model_name} already exists at {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading WAN model: {model_name}")
        try:
            # Use huggingface-cli download with --local-dir to properly handle sharded models
            cmd = [
                "hf", "download",
                model_name, 
                "--local-dir", str(model_path)
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout:
                logger.info(f"Download output: {result.stdout}")
            
            logger.info(f"Successfully downloaded {model_name} to {model_path}")
            return str(model_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {model_name}: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise
    
    def download_civitai_lora(self, url, filename=None):
        """Download LoRA from CivitAI URL"""
        try:
            # Extract model version ID from URL and convert to download API URL
            if "civitai.com" in url:
                download_url = url
                model_id = None
                
                # Handle different CivitAI URL formats
                if "modelVersionId=" in url:
                    # Extract model version ID from URL parameter
                    model_version_id = url.split("modelVersionId=")[1].split("&")[0]
                    download_url = f"https://civitai.com/api/download/models/{model_version_id}"
                    model_id = model_version_id
                elif "/models/" in url and "?modelVersionId=" in url:
                    # Handle full model page URLs like: https://civitai.com/models/1307155/wan-general-nsfw-model-fixed?modelVersionId=1475095
                    model_version_id = url.split("?modelVersionId=")[1].split("&")[0]
                    download_url = f"https://civitai.com/api/download/models/{model_version_id}"
                    model_id = model_version_id
                elif "/api/download/models/" in url:
                    model_id = url.split("/api/download/models/")[1].split("?")[0]
                    download_url = url
                else:
                    raise ValueError(f"Cannot parse CivitAI URL: {url}. Please use direct download URL or model page with modelVersionId parameter.")
                
                if not filename:
                    filename = f"lora_{model_id}.safetensors"
                
                lora_path = self.loras_dir / filename
                
                if lora_path.exists():
                    logger.info(f"LoRA already exists at {lora_path}")
                    return str(lora_path)
                
                logger.info(f"Downloading LoRA from: {download_url}")
                
                # Download with proper headers for CivitAI
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                # Add CivitAI API key if provided
                civitai_api_key = os.getenv("CIVITAI_API_KEY")
                if civitai_api_key:
                    headers["Authorization"] = f"Bearer {civitai_api_key}"
                    logger.info("Using CivitAI API key for authenticated download")
                else:
                    logger.warning("No CivitAI API key provided. Some models may require authentication.")
                    logger.info("Set CIVITAI_API_KEY environment variable for authenticated downloads.")
                
                response = requests.get(download_url, headers=headers, stream=True, allow_redirects=True)
                response.raise_for_status()
                
                # Check if we got HTML instead of binary file (authentication required)
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    error_msg = f"Received HTML instead of binary file from {download_url}"
                    if not civitai_api_key:
                        error_msg += "\n🔐 This model likely requires authentication. Please set CIVITAI_API_KEY environment variable."
                        error_msg += "\n📝 Get your API key from: https://civitai.com/user/account"
                    else:
                        error_msg += "\n❌ Authentication failed or model not accessible with provided API key."
                    raise ValueError(error_msg)
                
                # Check for proper file extension in response - but don't change path if file already exists
                content_disposition = response.headers.get('content-disposition', '')
                if content_disposition and 'filename=' in content_disposition:
                    # Extract filename from Content-Disposition header
                    suggested_filename = content_disposition.split('filename=')[1].strip('"')
                    if suggested_filename.endswith(('.safetensors', '.ckpt', '.bin', '.pth')):
                        # Create new path with suggested filename
                        suggested_lora_path = self.loras_dir / suggested_filename
                        # If the suggested file already exists, use that instead
                        if suggested_lora_path.exists():
                            logger.info(f"LoRA already exists with suggested filename at {suggested_lora_path}")
                            return str(suggested_lora_path)
                        # Only update filename and path if the suggested file doesn't exist
                        filename = suggested_filename
                        lora_path = suggested_lora_path
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify the downloaded file is not HTML
                file_size = lora_path.stat().st_size
                if file_size < 1024 * 100:  # File smaller than 100KB, might be error page
                    with open(lora_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1024)
                        if '<html' in content.lower() or '<!doctype html' in content.lower() or 'sign in' in content.lower():
                            lora_path.unlink()  # Delete the HTML file
                            error_msg = f"Downloaded HTML authentication page instead of LoRA file from {download_url}"
                            if not civitai_api_key:
                                error_msg += "\n🔐 This model requires authentication. Please set CIVITAI_API_KEY environment variable."
                                error_msg += "\n📝 Get your API key from: https://civitai.com/user/account"
                            raise ValueError(error_msg)
                
                logger.info(f"Successfully downloaded LoRA to {lora_path} ({file_size} bytes)")
                return str(lora_path)
            else:
                raise ValueError(f"Unsupported LoRA URL format: {url}")
                
        except Exception as e:
            logger.error(f"Failed to download LoRA from {url}: {e}")
            logger.info("💡 Tips for CivitAI LoRA downloads:")
            logger.info("   📋 Use direct download URLs: https://civitai.com/api/download/models/MODEL_VERSION_ID")
            logger.info("   📋 Or model page URLs: https://civitai.com/models/MODEL_ID/model-name?modelVersionId=VERSION_ID")
            logger.info("   🔐 Set CIVITAI_API_KEY for authenticated downloads: https://civitai.com/user/account")
            logger.info("   🔧 Alternatively, download manually and place in /runpod-volume/loras/")
            # Don't re-raise - let the process continue without this LoRA
            return None
    
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
                    if lora_path:  # Only add if download was successful
                        lora_paths.append(lora_path)
                        logger.info(f"✅ LoRA {i+1} downloaded successfully")
                    else:
                        logger.warning(f"⚠️ LoRA {i+1} download failed - continuing without it")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download LoRA {i+1} from {url}: {e}")
                    logger.warning(f"⏩ Continuing without LoRA {i+1}...")
        
        return lora_paths
    
    def setup_models(self):
        """Setup all required models and LoRAs"""
        logger.info("Setting up WAN models and LoRAs...")
        
        # Get model type from environment variable
        model_type = os.getenv("WAN_MODEL_TYPE", "TI2V-5B")
        
        if model_type == "TI2V-5B":
            model_name = "Wan-AI/Wan2.2-TI2V-5B"
        elif model_type == "T2V-A14B":
            model_name = "Wan-AI/Wan2.2-T2V-A14B"
        elif model_type == "I2V-A14B":
            model_name = "Wan-AI/Wan2.2-I2V-A14B"
        elif model_type == "I2V-14B-720P":
            model_name = "Wan-AI/Wan2.2-I2V-14B-720P"
        elif model_type == "I2V-14B-480P":
            model_name = "Wan-AI/Wan2.2-I2V-14B-480P"
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