# Hardware and memory

The PyTorch pipeline accepts `--device cuda`, `mps`, `xpu`, or `cpu`. AMD ROCm
uses PyTorch's `cuda` device name. `auto` selects CUDA/ROCm, MPS, XPU, then CPU.
The model uses BF16 and the audio decoder uses FP32. PyTorch is pinned to
2.14.1 on macOS and 2.10.0 elsewhere. Install the wheel appropriate for your
device before `pip install .`.

| Platform | Execution |
| --- | --- |
| NVIDIA, Linux or Windows | CUDA graphs when enabled; attention selected from available FlashAttention, cuDNN, and public SDPA kernels |
| AMD ROCm | Eager AR with public SDPA; no NVIDIA private attention operators |
| Apple Silicon | Eager AR, bucketed decode cache and short-query GQA folding |
| Intel XPU | Eager AR by default, device-local sampling and synchronization; optional compiled decode |
| CPU | Eager reference path |

These additional hardware paths are experimental. CPU tests cover equations,
cache visibility, placement order, cancellation/retry and artifact integrity.
They do not establish accelerator speed, supported song lengths or unchanged
full-model audio quality. Native Windows, MPS, AMD and XPU acceptance is pending.

## Installation

Use a fresh Python 3.12 environment. The [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/#v2100)
lists the matching driver/runtime requirements. Install **one** of these wheels:

```bash
# NVIDIA, Linux or Windows; choose a CUDA build compatible with your driver.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# AMD, Linux with a supported ROCm 7.1 GPU/runtime.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1

# Apple Silicon, macOS 14 or later.
python -m pip install torch==2.14.1

# Intel, with the Intel GPU driver installed.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/xpu
```

Then run `python -m pip install .` and `yue2 doctor`. For Intel driver and
platform requirements, see the [PyTorch 2.10 XPU guide](https://docs.pytorch.org/docs/2.10/notes/get_start_xpu.html).
Availability reported by `doctor` is an environment check; verify an actual
generation separately. AMD Windows wheels and `gfx1151` require a compatible
vendor runtime and are not covered by the Linux wheel recipe above.

The macOS pin avoids the older MPS attention errors reported in
[#176](https://github.com/multimodal-art-projection/YuE/issues/176) and
[#268](https://github.com/multimodal-art-projection/YuE/issues/268): causal
prefill leakage and incorrectly typed long-decode scratch buffers. See the
[upstream causal-attention report](https://github.com/pytorch/pytorch/issues/195910).
Native Mac generation acceptance on this revision is still pending. The
optional `fast` extra is for the NVIDIA environment, not macOS.

On NVIDIA builds without compiled FlashAttention, shared attention expands
grouped KV heads so PyTorch can use memory-efficient SDPA instead of its math
fallback ([#209](https://github.com/multimodal-art-projection/YuE/issues/209)).
Changing an attention kernel can change sampled outputs; compare within the
same runtime and backend when checking reproducibility.

## Optional compiled decode

Set `YUE2_COMPILE=1` to compile one-token AR decoding with `torch.compile`.
It is off by default and is ignored while CUDA graphs are active. The first
call compiles; later calls reuse the model-owned callable. Compilation can
change sampled outputs, so compare it separately from the eager baseline.
Native full-song testing currently covers Intel Arc Pro B70 with torch
2.10.0+xpu ([results](https://github.com/multimodal-art-projection/YuE/pull/255#issuecomment-6552181623)).

## Low memory

```bash
yue2 generate --request examples/song.json --device cuda --low-vram --budget 8
```

Python uses `YuE2Pipeline.from_pretrained(..., low_vram=True)`. This opt-in mode
keeps only the active AR or NAR modules on the accelerator, then unloads the
song model before VAE decoding. Each acoustic chunk prefills with AR weights,
switches to NAR weights, and releases its cache before the next chunk. FP8 AR
weights, when explicitly requested on supported NVIDIA GPUs, are restored to
their original BF16 values before acoustic prefill.

Transfers require host RAM and can reduce speed. CUDA/ROCm `--budget` retains
the existing 2 GiB reserve and allocator limit; low-memory mode never raises
it. This allocator limit does not cap driver memory or other processes. MPS
and XPU do not currently have an enforced allocator budget here. Stable length
limits on 6/8/12 GB devices remain to be measured; long KV caches can still
exhaust memory. The default is `low_vram=False`. Use the PyTorch backend for
this mode; the optional vLLM worker manages its own placement.

For AMD convolution startup tuning, an optional shell setting is
`MIOPEN_FIND_MODE=FAST`; see [MIOpen's find modes](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html).
It changes solver selection and needs its own timing and output comparison.
YuE2 does not set process-wide MIOpen, AOTriton, or GPU architecture overrides.

## Checking a platform

Run `python -m pytest tests/test_attention_portability.py tests/test_nar.py
tests/test_cuda_graph.py tests/test_devices.py`. Hardware-specific tests skip
when that device is absent. MPS checks include causal visibility and a fresh
process's first long decode. Compare fixed-input logits and latents before
comparing full songs. Measure peak device and driver memory, multiple acoustic
chunks, consecutive requests, cancellation and retry. Retain a 24 GB NVIDIA
default-path comparison when validating smaller devices.
