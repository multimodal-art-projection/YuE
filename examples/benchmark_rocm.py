"""Manual, single-process ROCm probes; no model downloads or service changes.

Run on an idle GPU. Attention uses synthetic tensors. Decode reuses your own
saved latents and local listening VAE. Timings exclude model loading.
"""
import argparse
import json
import time

import numpy as np
import torch

from yue2.nar import attention
from yue2.rocm import configure_profile, decoder_execution, runtime_config


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def attention_probe(device):
    configure_profile("windows-gfx1151", device, backend="torch", quantization="none")
    emit(runtime=runtime_config(device, "windows-gfx1151"))
    rng = torch.Generator(device="cpu").manual_seed(20260916)
    for qlen, klen, causal in [(1, 197, False), (1, 1389, False), (1, 12000, False),
                              (197, 197, True), (1389, 1389, True),
                              (128, 4443, False), (64, 24000, False)]:
        q = torch.randn((qlen, 16, 128), generator=rng).to(device=device, dtype=torch.bfloat16)
        k = torch.randn((klen, 8, 128), generator=rng).to(device=device, dtype=torch.bfloat16)
        v = torch.randn((klen, 8, 128), generator=rng).to(device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        expected = attention(q, k, v, causal=causal, backend="math")
        actual = attention(q, k, v, causal=causal, backend="flash")
        repeat = attention(q, k, v, causal=causal, backend="flash")
        torch.cuda.synchronize(device)
        delta = actual.float() - expected.float()
        relative = (delta.square().mean().sqrt() / expected.float().square().mean().sqrt()).item()
        maximum = delta.abs().max().item()
        finite = torch.isfinite(actual).all().item()
        exact = torch.equal(actual, repeat)
        emit(query=qlen, keys=klen, causal=causal, relative_rms=relative,
             max_abs=maximum, finite=finite, repeat_exact=exact,
             seconds=time.perf_counter() - start)
        if not (finite and exact and relative < 0.015 and maximum < 0.02):
            raise RuntimeError("Flash/math comparison exceeded the BF16 probe tolerance")
    emit(passed=True, cases=7)


def decode_probe(args, device):
    from yue2.modeling_vae import YuE2VAE

    if args.frames < 0 or args.repeats < 1:
        raise ValueError("frames must be nonnegative and repeats positive")
    profile = "windows-gfx1151" if args.decoder == "native" else None
    configure_profile(profile, device, backend="torch", quantization="none")
    latent = np.load(args.latents, allow_pickle=False)
    if latent.ndim != 2 or latent.shape[1] != 64 or len(latent) == 0 or not np.isfinite(latent).all():
        raise ValueError("Expected finite, nonempty saved latents [T,64]")
    if args.frames:
        latent = latent[:args.frames]
    model = YuE2VAE.from_pretrained(args.vae, decoder_only=True, device="cpu", local_files_only=True)
    model.to(device)
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("This probe requires the FP32 listening VAE")
    z = torch.as_tensor(latent, dtype=torch.float32).T.unsqueeze(0)
    emit(runtime=runtime_config(device, profile), frames=len(latent), decoder=args.decoder,
         core_frames=1024, halo_frames=16, model_dtype="float32")
    previous = None
    try:
        for index in range(args.repeats):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with decoder_execution(profile):
                audio = model.decode_tiled(z, core_frames=1024, halo_frames=16, output_device="cpu")
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            if not torch.isfinite(audio).all():
                raise RuntimeError("Decoder produced non-finite audio")
            emit(call=index + 1, seconds=seconds, samples=audio.shape[-1],
                 peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                 rms=audio.float().square().mean().sqrt().item(),
                 repeat_exact=None if previous is None else torch.equal(audio, previous))
            previous = audio
    finally:
        model.to("cpu")
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("attention")
    decode = sub.add_parser("decode")
    decode.add_argument("--vae", required=True, help="Local listening VAE directory")
    decode.add_argument("--latents", required=True, help="Saved latent.npy from save_artifacts")
    decode.add_argument("--decoder", choices=("upstream", "native"), required=True)
    decode.add_argument("--frames", type=int, default=128, help="0 explicitly selects the full sequence")
    decode.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This probe requires a ROCm GPU; use device cuda:N")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        if args.command == "attention":
            attention_probe(device)
        else:
            decode_probe(args, device)


if __name__ == "__main__":
    main()
