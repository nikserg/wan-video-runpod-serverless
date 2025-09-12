import subprocess
import os
import torch


# Check Pytorch CUDA availability
def _get_cuda_versions():
    versions = []
    # Попытка получить версию через nvcc
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        versions.append(output.decode())
    except Exception:
        # Проверка стандартных путей установки CUDA
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-*"
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                versions.append(f"Найдена CUDA по пути: {path}")
    return versions if versions else ["CUDA не найдена"]

def is_cuda_available():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        print("Доступные версии CUDA:")
        for v in _get_cuda_versions():
            print(v)
        return False
    else:
        current_cuda_capability = torch.cuda.get_device_capability()
        print(f"CUDA is available. Current GPU CUDA capability: {current_cuda_capability}")
        return True