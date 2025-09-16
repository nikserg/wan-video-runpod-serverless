import os
import sys
import subprocess
import tempfile
import base64
import logging
import json
import time
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIVideoGenerator:
    """Video generator using WAN CLI interface with torchrun"""

    def __init__(self, model_path, model_type="I2V-14B-480P", device="cuda", wan_repo_path="/wan"):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device
        self.wan_repo_path = Path(wan_repo_path)
        self.temp_dir = Path("/tmp/wan_generation")
        self.temp_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized CLI video generator for {model_type}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"WAN repo path: {self.wan_repo_path}")

    def get_task_from_model_type(self, model_type):
        """Map model type to CLI task parameter"""
        task_mapping = {
            "I2V-14B-480P": "i2v-A14B",
            "I2V-14B-720P": "i2v-A14B",
            "TI2V-5B": "ti2v-5B",
            "T2V-A14B": "t2v-A14B",
            "I2V-A14B": "i2v-A14B",
            "VACE-1.3B": "ti2v-5B"
        }
        return task_mapping.get(model_type, "i2v-A14B")

    def get_optimal_size_for_model(self, model_type, width=None, height=None):
        """Get optimal size for specific model"""
        if model_type == "I2V-14B-480P":
            return "832*480"
        elif model_type == "I2V-14B-720P":
            return "1280*720"
        elif model_type == "VACE-1.3B":
            return "832*480"
        else:
            # For other models, use provided dimensions or default
            if width and height:
                return f"{width}*{height}"
            return "1280*720"

    def save_image_to_temp(self, image_data, filename="input_image.jpg"):
        """Save base64 image data to temporary file"""
        temp_image_path = self.temp_dir / filename

        if isinstance(image_data, str):
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data)
            with open(temp_image_path, 'wb') as f:
                f.write(image_bytes)
        elif isinstance(image_data, bytes):
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        elif hasattr(image_data, 'save'):
            # PIL Image
            image_data.save(temp_image_path)
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")

        logger.info(f"Saved input image to {temp_image_path}")
        return str(temp_image_path)

    def check_gpu_count(self):
        """Check available GPU count"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line.startswith('GPU')])
            logger.info(f"Detected {gpu_count} GPU(s)")
            return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not detect GPU count, defaulting to 1")
            return 1

    def build_generation_command(self, task, size, image_path, prompt, negative_prompt="",
                               num_inference_steps=50, guidance_scale=5.0, seed=None,
                               use_multi_gpu=True):
        """Build the CLI command for video generation"""

        # Determine output file path using WAN's naming convention
        timestamp = int(time.time())
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        gpu_count = self.check_gpu_count() if use_multi_gpu else 1
        ulysses_size = min(gpu_count, 8) if use_multi_gpu and gpu_count > 1 else 1

        # Create save file path in temp directory for better control
        save_filename = f"{task}_{size}_{ulysses_size}_{formatted_prompt}_{timestamp}.mp4"
        save_file_path = self.temp_dir / save_filename

        # Base command
        if gpu_count > 1 and use_multi_gpu:
            # Multi-GPU with torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={ulysses_size}",  # Limit to 8 GPUs max
                str(self.wan_repo_path / "generate.py")
            ]
            # Add FSDP flags for multi-GPU
            multi_gpu_args = [
                "--dit_fsdp",
                "--t5_fsdp",
                f"--ulysses_size={ulysses_size}"
            ]
        else:
            # Single GPU
            cmd = ["python", str(self.wan_repo_path / "generate.py")]
            multi_gpu_args = []

        # Core arguments - simplified to match official example
        args = [
            "--task", task,
            "--size", size,
            "--ckpt_dir", str(self.model_path),
            "--image", image_path,
            "--prompt", prompt,
            "--save_file", str(save_file_path),
            "--offload_model", "True",
            "--convert_model_dtype"
        ]

        # Optional arguments
        if negative_prompt:
            args.extend(["--negative_prompt", negative_prompt])

        if seed is not None:
            args.extend(["--seed", str(seed)])

        if num_inference_steps != 50:
            args.extend(["--num_inference_steps", str(num_inference_steps)])

        if guidance_scale != 5.0:
            args.extend(["--guidance_scale", str(guidance_scale)])

        # Combine all arguments
        full_cmd = cmd + args + multi_gpu_args

        logger.info(f"Generation command: {' '.join(full_cmd)}")
        logger.info(f"Output will be saved to: {save_file_path}")
        return full_cmd, save_file_path

    def run_generation(self, cmd, save_file_path, timeout=600):
        """Run the generation command with proper error handling"""
        try:
            logger.info("Starting video generation...")

            # Set environment variables for GPU optimization and debugging
            env = os.environ.copy()
            env.update({
                'CUDA_VISIBLE_DEVICES': '0' if self.check_gpu_count() == 1 else ','.join(map(str, range(self.check_gpu_count()))),
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                'TOKENIZERS_PARALLELISM': 'false',  # Avoid tokenizer warnings
                'CUDA_LAUNCH_BLOCKING': '1',  # Synchronous CUDA for better error reporting
                'TORCH_USE_CUDA_DSA': '1'  # Enable device-side assertions for debugging
            })

            # Change to WAN repository directory
            result = subprocess.run(
                cmd,
                cwd=str(self.wan_repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=True
            )

            logger.info("✅ Video generation completed successfully")
            if result.stdout:
                logger.info(f"Generation output: {result.stdout}")

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Generation timed out after {timeout} seconds")
            raise RuntimeError(f"Video generation timed out after {timeout} seconds")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Generation failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Video generation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during generation: {e}")
            raise

    def find_output_video(self, task, size, prompt, before_time):
        """Find the generated video file using WAN's naming convention"""
        # WAN saves in the working directory (wan repo path) with this pattern:
        # {task}_{size}_{ulysses_size}_{formatted_prompt}_{timestamp}.mp4

        search_dir = self.wan_repo_path
        video_files = list(search_dir.glob("*.mp4"))

        if not video_files:
            return None

        # Filter for videos created after our generation started
        recent_videos = []
        for video_file in video_files:
            file_mtime = video_file.stat().st_mtime
            if file_mtime > before_time:
                recent_videos.append(video_file)

        if not recent_videos:
            # Fallback: return most recent video file
            return max(video_files, key=lambda p: p.stat().st_mtime) if video_files else None

        # Look for video that matches our task and size pattern
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        size_formatted = size.replace("*", "x")

        for video_file in recent_videos:
            filename = video_file.name
            # Check if filename starts with expected pattern
            if filename.startswith(f"{task}_{size}") or filename.startswith(f"{task}_{size_formatted}"):
                return video_file

        # Return most recent video if no pattern match
        return max(recent_videos, key=lambda p: p.stat().st_mtime)

    def video_to_base64(self, video_path):
        """Convert video file to base64 string"""
        try:
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')

            logger.info(f"Converted video to base64: {len(video_bytes)} bytes")
            return video_base64
        except Exception as e:
            logger.error(f"Failed to convert video to base64: {e}")
            raise

    def cleanup_temp_files(self, *paths):
        """Clean up temporary files"""
        for path in paths:
            try:
                if isinstance(path, (str, Path)):
                    path = Path(path)
                    if path.exists():
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        logger.debug(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")

    def generate_video(self,
                      prompt,
                      image=None,
                      negative_prompt="",
                      width=832,
                      height=480,
                      duration_seconds=5.0,
                      fps=24,
                      guidance_scale=5.0,
                      num_inference_steps=50,
                      seed=None,
                      resolution_preset=None,
                      use_multi_gpu=None):
        """Generate video using WAN CLI interface"""

        temp_image_path = None
        save_file_path = None

        try:
            # Validate inputs
            if not prompt:
                raise ValueError("Prompt is required")

            if image is None:
                raise ValueError(f"Model {self.model_type} requires an input image")

            # Auto-detect multi-GPU usage based on available GPUs
            if use_multi_gpu is None:
                gpu_count = self.check_gpu_count()
                use_multi_gpu = gpu_count > 1
                logger.info(f"Auto-detected multi-GPU usage: {use_multi_gpu} ({gpu_count} GPUs)")

            # Get task and size for model
            task = self.get_task_from_model_type(self.model_type)
            size = self.get_optimal_size_for_model(self.model_type, width, height)

            logger.info(f"Using task: {task}, size: {size}")

            # Save input image to temporary file
            temp_image_path = self.save_image_to_temp(image)

            # Record time before generation for output detection
            before_time = time.time()

            # Build generation command with save_file path
            cmd, save_file_path_result = self.build_generation_command(
                task=task,
                size=size,
                image_path=temp_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                use_multi_gpu=use_multi_gpu
            )

            # Store the save file path for cleanup
            save_file_path = save_file_path_result

            # Run generation
            self.run_generation(cmd, save_file_path)

            # Check if the specified output file was created
            if not save_file_path.exists():
                raise RuntimeError(f"Video file not found at expected location: {save_file_path}")

            logger.info(f"Generated video: {save_file_path}")
            video_path = save_file_path

            # Convert to base64
            video_base64 = self.video_to_base64(video_path)

            return video_base64

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
        finally:
            # Cleanup temporary files
            if temp_image_path:
                self.cleanup_temp_files(temp_image_path)
            if save_file_path:
                self.cleanup_temp_files(save_file_path)


class MockCLIVideoGenerator:
    """Mock CLI video generator for testing"""

    def __init__(self, model_path, model_type="I2V-14B-480P", device="cuda", wan_repo_path="/wan"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        logger.info(f"Using mock CLI video generator for {model_type}")

    def generate_video(self, prompt, image=None, duration_seconds=5.0, fps=24, **kwargs):
        """Generate a mock video"""
        try:
            logger.info(f"Mock CLI generation: {prompt[:50]}...")

            # Create simple colored frames
            width, height = 832, 480
            num_frames = min(int(duration_seconds * fps), 120)

            frames = []
            for i in range(num_frames):
                color = (i * 8 % 255, (i * 16) % 255, (i * 32) % 255)
                frame = np.full((height, width, 3), color, dtype=np.uint8)

                # Add CLI-specific text
                cv2.putText(frame, f"CLI Mock Gen Frame {i+1}", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, f"{self.model_type}", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, prompt[:40], (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                frames.append(frame)

            return self.frames_to_mp4(frames, fps)

        except Exception as e:
            logger.error(f"Mock CLI generation failed: {e}")
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

            logger.info(f"Generated mock CLI MP4 video: {len(video_bytes)} bytes")
            return video_base64

        except Exception as e:
            logger.error(f"Failed to convert mock frames to MP4: {e}")
            raise


def create_cli_generator(model_path, model_type="I2V-14B-480P", device="cuda", use_mock=False, wan_repo_path="/wan"):
    """Factory function to create CLI video generator"""
    if use_mock:
        return MockCLIVideoGenerator(model_path, model_type, device, wan_repo_path)
    else:
        return CLIVideoGenerator(model_path, model_type, device, wan_repo_path)