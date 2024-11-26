import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Get the version of CUDA
cuda_version = torch.version.cuda

# Get the name of the GPU (if available)
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(f"GPU Name: {gpu_name}")
