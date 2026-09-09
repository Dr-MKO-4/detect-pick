"""Hardware detection  runs once at import time, exposes DEVICE and AMP_ENABLED."""
import os
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    AMP_ENABLED = True
    torch.backends.cuda.matmul.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    AMP_ENABLED = False
else:
    DEVICE = torch.device("cpu")
    AMP_ENABLED = False
    torch.set_num_threads(os.cpu_count() or 1)
