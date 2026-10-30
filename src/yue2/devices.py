"""Device selection and request-boundary cleanup for the PyTorch pipeline."""
import torch


def resolve_device(device):
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else
                  "xpu" if torch.xpu.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda", "mps", "xpu"}:
        raise ValueError("device must be cpu, cuda, mps, xpu, or auto")
    if device.type in {"cuda", "xpu"}:
        backend = getattr(torch, device.type)
        if not backend.is_available():
            raise RuntimeError(f"{device.type} is unavailable in this PyTorch installation")
        if device.index is None:
            device = torch.device(device.type, backend.current_device())
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch installation")
    return device


def synchronize(device):
    device = torch.device(device)
    if device.type in {"cuda", "xpu"}:
        getattr(torch, device.type).synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device):
    device = torch.device(device)
    if device.type in {"cuda", "xpu"}:
        backend = getattr(torch, device.type)
        with backend.device(device):
            backend.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
