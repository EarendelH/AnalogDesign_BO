import torch
import ray

gpu_available = torch.cuda.is_available()
print(f"Is CUDA GPU available: {gpu_available}")
if gpu_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA GPUs available: {num_gpus}")

ray.init()

resources = ray.available_resources()
num_gpus_ray = resources.get("GPU", 0)
print(f"Number of GPUs Ray can use: {num_gpus_ray}")

ray.shutdown()
