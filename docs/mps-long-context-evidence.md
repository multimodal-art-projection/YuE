# Apple MPS long-context decode evidence

This note records the reproducible evidence behind the Apple MPS fix in
[PR #187](https://github.com/multimodal-art-projection/YuE/pull/187). It is
intentionally separate from the implementation so reviewers can distinguish
unit-test coverage, synthetic reproduction, and real-checkpoint observations.

## Revision and environments

Implementation under review:

- YuE commit: `8db0b9b4a058840ad1a2e71c1fffd645ae081de2`
- Model revision: `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`
- VAE revision: `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`

Synthetic benchmark host:

- Apple M4 Max, 128 GB, arm64 (`Mac16,5`)
- macOS 15.7.9 (24G830)
- Python 3.12.9
- PyTorch 2.10.0
- Transformers 4.57.6
- MPS, float16

Real-checkpoint host:

- Apple M5 Max, 128 GB, arm64 (`Mac17,6`)
- macOS 26.5.1 (25F80)
- Python 3.12.9
- PyTorch 2.10.0
- Transformers 4.57.6
- MPS, bfloat16 model weights
- `YUE2_MPS_EMPTY_CACHE=0`

## Test coverage

The complete offline suite at the reviewed commit reports:

```text
187 passed, 11 skipped, 30 subtests passed in 3.31s
```

The PR adds 11 test functions (13 tests/subtests after parametrization):

1. Bucket boundaries and the capacity cap.
2. Monotonic bucket transitions.
3. Exact, unbucketed prefix views on non-MPS devices.
4. Broadcast GQA equivalence to explicit K/V duplication across several head ratios and lengths.
5. Causal broadcast GQA equivalence.
6. Unused-slot masking equivalence to shortened K/V tensors.
7. Full decode equivalence across several forced bucket transitions.
8. Padded decode retaining an exact prefix rather than an incompatible bucket.
9. MPS GQA equivalence to an FP32 CPU reference for float16, bfloat16, and float32.
10. MPS causal attention not leaking future keys.
11. Real-MPS bucketed decode equivalence to an uncached full-prefix forward pass.

Tests 1–8 also run in CPU CI. Tests 9–11 require Apple MPS and therefore run only
on an Apple host. The upstream GitHub workflow currently has no check run for
this fork PR; the local result is not presented as a substitute for maintainer-
authorized CI.

Reproduce locally:

```text
python -m pytest -q
```

## Checkpoint-free synthetic reproduction

The included benchmark launches the growing and bucketed policies in separate
processes to avoid sharing a Metal graph cache:

```text
python tools/mps_long_context_bench.py --mode both --tokens 2048
python tools/mps_long_context_bench.py --mode both --tokens 4096
```

Raw machine-readable output is committed in
[`mps-long-context-synthetic.json`](evidence/mps-long-context-synthetic.json).

| Requested decode tokens | Policy | Distinct K/V lengths | Peak allocated GiB | Peak driver GiB | Tokens/s |
|---:|---|---:|---:|---:|---:|
| 2,048 | growing prefix | 2,048 | 0.05 | 0.24 | 196.62 |
| 2,048 | bucketed | 2 | 0.05 | 0.10 | 182.16 |
| 4,096 | growing prefix | 4,096 | 0.00 | 0.09 | 151.58 |
| 4,096 | bucketed | 3 | 0.00 | 0.03 | 344.14 |

The robust observation is shape count: thousands of decode shapes collapse to
two or three. Absolute memory and throughput depend on model size and system
state, so the small synthetic memory figures should not be extrapolated to the
3B checkpoint.

## Real 3B checkpoint

### Controlled failure progression

Before bucketing, a diagnostic revision retained the allocation-free GQA change
but returned the exact growing K/V prefix. PyTorch tensor allocation remained
7.80 GiB while Metal driver allocation climbed sharply:

| Semantic tokens | PyTorch allocated GiB | Metal driver GiB |
|---:|---:|---:|
| 5,632 | 7.80 | 9.41 |
| 6,400 | 7.80 | 16.41 |
| 7,424 | 7.80 | 43.41 |
| 8,192 | 7.80 | 68.41 |

This is an isolation experiment, not a benchmark of unmodified `main`: only the
cache shape policy differs from the fixed-shape diagnostic path. It demonstrates
that live tensor size does not explain the driver growth.

### Reviewed implementation with print-only sampling

The reviewed `modeling_yue2.py` was staged byte-for-byte from commit `8db0b9b`.
For this run only, print-only instrumentation sampled
`torch.mps.current_allocated_memory()` and
`torch.mps.driver_allocated_memory()` every 256 semantic tokens. The
instrumentation did not clear caches or alter attention/cache behavior.

The complete trace is committed as
[`mps-long-context-real-memory.csv`](evidence/mps-long-context-real-memory.csv), and the
growing-prefix diagnostic trace is in
[`mps-long-context-growing-memory.csv`](evidence/mps-long-context-growing-memory.csv).
Its key points are:

| Semantic tokens | PyTorch allocated GiB | Metal driver GiB |
|---:|---:|---:|
| 256 | 8.10 | 9.42 |
| 4,096 | 8.10 | 9.42 |
| 4,352 | 8.10 | 9.42 |
| 8,192 | 8.10 | 9.42 |
| 8,960 | 8.10 | 9.42 |

The exact command, after staging the reviewed `src/` tree and request on the M5 host, was:

```text
env PYTHONPATH=/Users/mister5/ActiveProjects/YuE2-mps-pr/src HF_HUB_OFFLINE=1 YUE2_MPS_MEMORY_LOG=1 YUE2_MPS_EMPTY_CACHE=0 python -m yue2.cli generate --offline --device mps --budget 24 --revision 29b3558dd46954a0cd9021dc76d5c91864a0f1c7 --vae-revision 9a94e1d0ea9f8087e98f77fa88df4a4068104d2a --request generation-request.json --output artifacts
```

`YUE2_MPS_MEMORY_LOG` enabled the print-only sampler in the staged validation
copy; it is not required by or included in this PR.

The run completed 9,000 semantic tokens and decoded 360.0 seconds of stereo
48 kHz FLAC. Driver allocation stayed flat across the 4,096 to 8,192 bucket
transition instead of increasing with every token.

A second isolated run with the exact submitted source and no instrumentation
completed:

- 13,500 semantic tokens in 563.60 seconds (23.95 tokens/s)
- 64 of 64 NAR synthesis steps
- 14 of 14 audio decoder chunks
- 540.0 seconds of stereo 48 kHz FLAC
- 1,647.88 seconds end to end

The compact machine-readable results are in
[`mps-long-context-real-runs.json`](evidence/mps-long-context-real-runs.json), and artifact
checksums are recorded in
[`mps-long-context-artifacts.sha256`](evidence/mps-long-context-artifacts.sha256).
Large generated audio and model weights are intentionally not committed.

### Earlier long-context observation

An earlier bucketed revision completed 18,000 semantic tokens and 720.0 seconds
of audio, with 10.75 GiB PyTorch allocation and 11.94 GiB Metal driver allocation
remaining flat through samples at 16,640 tokens. This supports the long-context
claim but is labeled separately because it predates the final cleanup commit.
The 9,000-token memory trace and 13,500-token end-to-end run above are the direct
validation of the submitted implementation.

## Interpretation and limitations

What the evidence directly establishes:

- Exact per-token K/V prefix lengths create thousands of MPS decode shapes.
- Bucketing bounds those lengths to a small set.
- Bucket masking and broadcast GQA are numerically equivalent in the tested cases.
- On the real checkpoint, driver memory stays flat across long decode and a bucket transition.
- The submitted implementation completes a 13,500-token, nine-minute end-to-end generation without cache clearing.

What it does not establish:

- A universal memory or throughput number for every Apple GPU and PyTorch release.
- A green upstream GitHub check; the workflow still awaits authorization/execution.
- Bitwise equivalence for every model input. The tests use tolerance-based numerical equivalence appropriate to SDPA.
- That the earlier 18,000-token result was produced by the final cleanup commit; it was produced by an earlier bucketed revision and is labeled accordingly.
