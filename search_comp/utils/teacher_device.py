"""Resolve a single teacher device per torchrun process, without auto sharding."""
from __future__ import annotations

import os
import re


def resolve_teacher_device(device="auto"):
    """CUDA ordinals are logical indices within CUDA_VISIBLE_DEVICES."""
    import torch

    if device == "cpu":
        return "cpu"
    if not isinstance(device, str) or not re.fullmatch(r"auto|cuda(?::[0-9]+)?", device):
        raise ValueError("distill_teacher_device must be auto, cpu, cuda or cuda:N")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and "LOCAL_RANK" not in os.environ:
        raise ValueError("Distributed teacher placement requires LOCAL_RANK; launch with torchrun")
    if not torch.cuda.is_available():
        if device == "auto" and not distributed:
            return "cpu"
        raise ValueError("GPU teacher requires CUDA; check CUDA_VISIBLE_DEVICES or explicitly select cpu")
    local_rank = int(os.environ["LOCAL_RANK"]) if distributed else torch.cuda.current_device()
    index = int(device.split(":")[1]) if ":" in device else local_rank
    if distributed and index != local_rank:
        raise ValueError(
            f"Teacher cuda:{index} conflicts with LOCAL_RANK={local_rank}; "
            "use auto or cuda to place each process on its own GPU"
        )
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"Teacher cuda:{index} is outside the {count} visible CUDA devices")
    return f"cuda:{index}"
