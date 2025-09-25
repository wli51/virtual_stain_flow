from functools import wraps

import torch

def with_cuda_device(method):
    """
    Decorator for Trainer methods that need to run on self.device.
    Ensures default CUDA device is set correctly for the duration.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        prev_device = torch.cuda.current_device()
        try:
            # Make sure self.device is a torch.device
            device = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
            torch.cuda.set_device(device)
            return method(self, *args, **kwargs)
        finally:
            torch.cuda.set_device(prev_device)
    return wrapper

class cuda_device:
    """
    Context manager for temporarily setting the default CUDA device.
    Ensures tensors created without explicit `device=` args go to the right GPU.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.prev = None

    def __enter__(self):
        self.prev = torch.cuda.current_device()
        torch.cuda.set_device(self.device)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.set_device(self.prev)