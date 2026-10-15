"""Explicit device/precision policy; unsupported requests fail before output creation."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Backend:
    device: torch.device
    amp: bool
    dtype: torch.dtype | None

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()


def resolve_backend(device: str, amp: bool = False) -> Backend:
    if device == "cpu":
        resolved = torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        resolved = torch.device("cuda")
    elif device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        resolved = torch.device("mps")
    else:
        raise ValueError("device must be cpu, cuda, or mps")
    if type(amp) is not bool:
        raise ValueError("amp must be a boolean")
    if amp:
        raise ValueError("AMP is disabled until CUDA mixed-precision behavior is explicitly validated")
    return Backend(resolved, False, None)
