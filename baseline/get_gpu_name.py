import torch

# Get the first visible GPU name
if torch.cuda.is_available():
    gpu_index = 0  # First GPU
    gpu_name = torch.cuda.get_device_name(gpu_index)
    print(gpu_name)
else:
    print("No CUDA device available")
