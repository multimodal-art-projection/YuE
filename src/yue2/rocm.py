"""Opt-in, hardware-scoped ROCm tuning; NVIDIA/CPU/MPS defaults are unchanged."""
from contextlib import contextmanager
import os
import sys
import warnings

import torch


AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
PROFILES = ("windows-gfx1151",)
TESTED_TORCH = "2.9.1+rocm7.2.1"


def configure_profile(profile, device, *, backend, quantization):
    """Configure before the first attention call, in a dedicated inference process.

    This deliberately does not enable experimental kernels on untested runtimes.
    An explicit environment opt-out (0) always wins over the profile default.
    """
    if profile is None:
        return
    if profile not in PROFILES:
        raise ValueError(f"Unknown ROCm profile: {profile}; choose from {PROFILES}")
    if device.type != "cuda" or not torch.version.hip:
        raise ValueError("The ROCm profile requires a ROCm GPU (device='cuda')")
    if backend not in {"torch", "torch-eager"} or quantization != "none":
        raise ValueError("The ROCm profile is validated only with unquantized torch/torch-eager")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":", 1)[0]
    if sys.platform != "win32" or str(torch.__version__) != TESTED_TORCH or arch != "gfx1151":
        raise ValueError(
            f"{profile} requires Windows, torch {TESTED_TORCH}, and gfx1151; "
            f"got {sys.platform}, torch {torch.__version__}, {arch or 'unknown architecture'}. "
            "Omit rocm_profile to use the untuned eager ROCm path."
        )
    os.environ.setdefault(AOTRITON_ENV, "1")
    warnings.warn(
        "Experimental windows-gfx1151 profile: eager AR, native FP32 GPU VAE, "
        f"{AOTRITON_ENV}={os.environ[AOTRITON_ENV]}. "
        "Use a fresh, single-request inference process; see docs/amd-rocm.md.",
        RuntimeWarning, stacklevel=2,
    )


@contextmanager
def decoder_execution(profile):
    """Bypass slow MIOpen convolutions without moving the FP32 VAE to CPU.

    cudnn.enabled also controls MIOpen dispatch in this PyTorch ROCm build.
    It is process-global: other model execution must not overlap this context.
    """
    if profile is None:
        yield
        return
    previous = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = False
        yield
    finally:
        torch.backends.cudnn.enabled = previous


def runtime_config(device, profile):
    """Record policy, not a claim that every SDPA call uses a Flash kernel."""
    if device.type != "cuda" or not torch.version.hip:
        return None
    props = torch.cuda.get_device_properties(device)
    return {
        "profile": profile,
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "architecture": getattr(props, "gcnArchName", None),
        "attention": "sdpa-auto",
        "experimental_aotriton": os.environ.get(AOTRITON_ENV),
        "vae_convolution": "torch-native" if profile is not None else "upstream",
    }
