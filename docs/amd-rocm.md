# Experimental Windows ROCm profile

YuE2 can use the AMD Ryzen AI Max+ 395's Radeon 8060S (`gfx1151`) with the
Windows ROCm PyTorch build described below. This is a measured configuration,
not a claim that every AMD GPU or ROCm version is supported. The normal NVIDIA
installation and CUDA-graph execution remain unchanged.

## Tested environment and installation

- Windows 11 Pro (build 26200), Python 3.12.10, `torch==2.9.1+rocm7.2.1`.
- HIP `7.2.53211-158bd99533`, Radeon 8060S (`gfx1151`).
- About 99.7 GiB GPU-visible shared memory on the test machine; a 32 GiB pipeline
  budget. **This is not a real-24-GB-device acceptance test.**
- BF16 model, FP32 listening VAE, no quantization or vLLM.
- Model `m-a-p/YuE2-3B`, revision `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`.
- VAE `m-a-p/YuE2-Vae`, revision `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`.

Create a **separate Python 3.12 environment** and follow AMD's
[ROCm 7.2.1 Windows installation instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html)
for the matching driver, runtime and PyTorch wheels. Verify `torch.version.hip`,
`torch.cuda.is_available()` and `torch.cuda.get_device_properties(0).gcnArchName`.
ROCm deliberately uses the
[`torch.cuda` interface and `cuda` device name](https://docs.pytorch.org/docs/2.9/notes/hip.html).

Do **not** run the default `pip install .` or install `.[fast]` into this AMD
environment: the normal release pins PyTorch 2.10.0. Keep the AMD wheel and
install the other dependencies explicitly, then the project without dependency
resolution (from the repository root):

```powershell
python -m pip install transformers==4.57.6 huggingface-hub==0.36.2 safetensors==0.7.0 tiktoken==0.12.0 numpy==2.2.6 soundfile==0.13.1 accelerate==1.13.0
python -m pip install --no-deps .
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```

`pip check` will report the deliberate difference from the package's NVIDIA
torch pin; this profile does not relax that default for other users. Future
dependency upgrades may overwrite the AMD wheel. Keep this environment isolated.

## Enable the profile

```powershell
python -X utf8 -m yue2.cli generate --request examples/song.json --device cuda --budget 32 --rocm-profile windows-gfx1151 --output outputs/amd-song
```

Or use `YuE2Pipeline.from_pretrained(..., device="cuda", memory_budget_gib=32,
rocm_profile="windows-gfx1151")`. The profile refuses untested OS, torch version
or GPU architecture, and refuses FP8/vLLM. The selected device is checked, not
unconditionally GPU 0. Omit the profile for untuned ROCm inference.

Use a **fresh process, with one request at a time**. Configure the profile before
any attention operation. PyTorch's experimental-kernel environment setting and
convolution backend flag are process-global; do not mix model services,
concurrent pipelines or differently configured GPUs in this process. The VAE
flag is restored on success, exception and cancellation. The AOTriton setting
is intentionally retained for the process lifetime: changing it after kernels
have initialized is not a reliable way to compare backends.

To retain the native VAE speedup but disable experimental attention, set
`$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "0"` **before launching Python**.
An explicit environment value is never overwritten. The profile's default is
`1`; PyTorch SDPA still selects a supported kernel for each operation. It does
not force every call through Flash attention or require installing Triton/vLLM.

Saved effective configuration includes the ROCm profile, runtime, architecture,
VAE convolution policy and attention environment setting. These enter the
request identity, so `--resume` cannot silently reuse a result from a different
profile. `validation_status` remains `unvalidated`; this is not reproduction of
the published quality benchmark.

## What changes, and why

1. **Use eager AR on ROCm.** The current CUDA-graph varlen attention path passes
   `seqused_k`. The tested ROCm operator advertises this argument but rejects it
   with `[ROCm] mha_varlen_fwd: seqused_k must be nullopt`. A full-capacity masked
   SDPA graph was slower than eager decoding, which slices the populated KV
   cache. ROCm therefore falls back to eager and records
   `rocm_not_graph_validated` in stage timing when graphs were requested.
2. **Use native GPU convolutions for the FP32 VAE.** In this ROCm build,
   `torch.backends.cudnn.enabled = False` bypasses MIOpen dispatch. The VAE and
   inputs remain on the GPU, at FP32. The change is scoped to decoding and
   restored in `finally`; it is not a CPU fallback. Initial MIOpen calls showed
   substantial overhead and warmed calls were still slower. Solver search,
   compilation and convolution time were not individually profiled, so the
   measurements do not assign all overhead to a single cause.
3. **Opt in to bundled experimental AOTriton kernels.** On this exact runtime
   and architecture, SDPA can use faster attention for AR and acoustic synthesis.
   No ODE steps, guidance, precision, model weights or sampling-quality settings
   are reduced by the profile.

## Measurements

Same synthetic request, seed `20260915`, full melody/chord planning, 32 midpoint
ODE steps, 1024-frame VAE cores and 16-frame halo. All three runs used eager AR;
the original default graph path failed on this runtime. The cue's explicit
semantic bounds were 1501..1551 tokens in **every** comparison. It reached the
1551-token cap, producing about 62.039 seconds of raw stereo audio; this is a
duration-bounded cue, **not a naturally completed song**. The downstream player
trimmed it to 60 seconds with a fade, which is not part of this patch or timing.

| Pipeline stage (seconds) | Original ROCm eager | Native VAE only | Native VAE + AOTriton |
|---|---:|---:|---:|
| Score planning | — | 98.778 | 80.468 |
| Semantic AR | — | 146.976 | 109.179 |
| Acoustic synthesis | — | 93.944 | 30.281 |
| VAE, including loading | 455.120 | 4.399 | 4.558 |
| Whole pipeline | 801.241 | 352.210 | 232.716 |

For this request the combined profile reduced measured pipeline time by about
**71% (3.44x throughput)**. Application-level wall time, including integrity
checks and publication, was 361.646 seconds for native VAE only and 242.169
seconds with both changes; those app timings are **not** upstream CLI timings.
These are individual observed runs, not an averaged benchmark across genres,
durations, cold caches or hardware. They came from the equivalent downstream
adapter on source `0edaf2f4053ef4731334b8329834b107977f9637`; the affected upstream
code was unchanged at the contribution base
`4d53bd5fc7e96a53cb907d3eb407a65df67a8b79`.

Controlled checks:

- Same 128-frame latent excerpt: MIOpen first/repeat calls **36.851/1.475 s**;
  native GPU **0.318/0.208 s**. A fresh MIOpen process after that cache warmed
  still took **12.646/1.510 s**. First-call costs are not universal constants.
- Same full 1551-frame latents: native decode-only **2.874/2.526 s**, about
  **5.53 GiB peak GPU allocation**. This excludes loading and should not be
  compared directly with the 455.120-second full VAE stage.
- Native vs original decoded waveform: RMS difference **2.72e-7**, max absolute
  difference **1.62e-5**, signal/difference ratio **116.165 dB**. GPU/FP32 placement
  was checked. This is numerical evidence, not a subjective listening study.
- Seven BF16/GQA attention cases (16 Q heads, 8 KV heads, head dimension 128)
  covered short/long AR, causal prefill and NAR shapes up to 24000 keys. Forced
  Flash vs math relative RMS was **0.00188..0.00254**, maximum absolute difference
  at most **0.015625**; samples were finite and repeated Flash output was exact.
- Two fresh optimized full runs had identical score, semantic tokens, latents
  and audio on this installation. **The same seed does not reproduce the old
  math-attention song** after changing attention arithmetic. Do not promise
  bitwise reproducibility across backends, software versions or devices.

NVIDIA hardware parity, other AMD architectures, Linux ROCm, newer runtimes,
FP8, vLLM, multi-GPU/concurrent operation and long-song quality were not tested.

The upstream-adapted profile was also run end to end on the same machine:
**228.794 seconds pipeline / 234.919 seconds total**. Its score, semantic-token
array, acoustic-latent array and raw audio FLAC were byte-identical to the
accepted optimized downstream run. Both AR stages reported the intended eager
ROCm fallback. The offline CPU suite passed **204 tests and 30 subtests**, with
15 platform/optional-GPU skips; the standalone wheel built successfully.

## Reproduce

Run on an idle GPU in fresh processes. Keep versions, power settings, model
revisions, input, seed and cache conditions fixed. Do not delete shared caches.
Retain `result.json`, `config.json`, tokens, latents and failures, not just audio.

```powershell
# Same fixed synthetic cue and sampling bounds as the measurements above.
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "0"
python -X utf8 -m yue2.cli generate --request examples/rocm-benchmark-song.json --backend torch-eager --budget 32 --revision 29b3558dd46954a0cd9021dc76d5c91864a0f1c7 --vae-revision 9a94e1d0ea9f8087e98f77fa88df4a4068104d2a --output outputs/rocm-baseline
python -X utf8 -m yue2.cli generate --request examples/rocm-benchmark-song.json --rocm-profile windows-gfx1151 --budget 32 --revision 29b3558dd46954a0cd9021dc76d5c91864a0f1c7 --vae-revision 9a94e1d0ea9f8087e98f77fa88df4a4068104d2a --output outputs/rocm-native
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
python -X utf8 -m yue2.cli generate --request examples/rocm-benchmark-song.json --rocm-profile windows-gfx1151 --budget 32 --revision 29b3558dd46954a0cd9021dc76d5c91864a0f1c7 --vae-revision 9a94e1d0ea9f8087e98f77fa88df4a4068104d2a --output outputs/rocm-tuned
```

Compare `timing` and `truncated` in each result. Re-run with fresh output
directories to assess variability; do not use `--resume` for timing. Add
`--model` and `--vae` with local directories plus `--offline` to reuse already
downloaded, revision-verified checkpoints.

The lightweight hardware probe needs no song weights:

```powershell
python -X utf8 examples/benchmark_rocm.py attention
```

For isolated VAE measurements, supply the listening VAE directory and a saved
`latent.npy`. Use separate processes for `--decoder upstream` and `native`.
The default 128-frame excerpt keeps MIOpen tests short; `--frames 0` opts into
the full latent sequence. Compare decode-only timing with decode-only timing.

```powershell
python -X utf8 examples/benchmark_rocm.py decode --vae models/YuE2-Vae --latents outputs/rocm-baseline/rocm-cue/latent.npy --decoder upstream
python -X utf8 examples/benchmark_rocm.py decode --vae models/YuE2-Vae --latents outputs/rocm-baseline/rocm-cue/latent.npy --decoder native
```

CPU policy tests require no weights or GPU: install `pytest==9.0.3` separately,
then `python -X utf8 -m pytest -q`. Linux-only vLLM transport/locking and
NVIDIA-only graph-capture tests are explicitly skipped on Windows/ROCm. The
default Linux/CPU CI continues to exercise those platform-appropriate tests.
