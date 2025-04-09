import torch
print("Torch version:", torch.__version__)
print("CUDA version bundled:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

