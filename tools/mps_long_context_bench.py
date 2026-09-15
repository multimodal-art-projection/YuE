#!/usr/bin/env python3
"""Bounded Apple MPS reproducer for YuE2 long-context autoregressive decode.

``StaticKVCache.update`` historically returned a K/V prefix whose sequence
length grew by exactly one at every decoded token::

    cache[:, :, :current_length]

On Apple MPS, Metal specializes kernel graphs per operand shape, so a
per-token-changing shape makes PyTorch allocate a fresh attention/matmul
workspace for nearly every token. Driver memory climbs far above the tensor
size itself, and long generations eventually stall or produce non-finite
values.

This script decodes the same synthetic model twice:

* ``grow``   -- the historical policy: an exact, ever-growing K/V prefix.
* ``bucket`` -- the fixed policy: power-of-two prefixes with the unused slots
                hidden by an attention mask, so only a handful of shapes exist.

It reports distinct decode shapes, peak PyTorch-allocated memory, peak Metal
driver memory, throughput, and whether the run completed. No checkpoint or
model download is required, and each result is printed as JSON.

Usage::

    python tools/mps_long_context_bench.py --mode both
    python tools/mps_long_context_bench.py --mode grow --tokens 4096

Run ``--mode both`` (the default) so that each policy gets a fresh process and
a fresh Metal driver-memory baseline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yue2.modeling_yue2 import StaticKVCache, YuE2Config, YuE2ForCausalLM  # noqa: E402

GIB = 2 ** 30


def build_model(args, device):
    config = YuE2Config(
        hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim, intermediate_size=args.hidden * 2,
        vocab_size=args.vocab,
        max_position_embeddings=args.prompt + args.tokens + 8,
        max_latent_frames=64, pad_token_id=0, eos_token_id=None, bos_token_id=1,
    )
    torch.manual_seed(0)
    model = YuE2ForCausalLM(config).eval().to(device=device, dtype=args.dtype)
    return model, config


def run_mode(args) -> dict:
    device = torch.device("mps")
    model, config = build_model(args, device)
    cache = StaticKVCache(
        num_layers=config.num_hidden_layers, batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        max_seq_len=args.prompt + args.tokens + 2,
        head_dim=config.head_dim, dtype=args.dtype, device=device,
    )
    cache.MPS_DECODE_BUCKET = args.bucket
    if args.mode == "grow":
        # Historical behaviour: never return a bucketed, fixed-shape prefix.
        cache._bucket_decode = lambda key_states: False

    seen_lengths: set[int] = set()
    original_update = StaticKVCache.update

    def instrumented_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = original_update(self, key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx == 0 and key_states.shape[2] == 1:
            seen_lengths.add(int(k.shape[2]))
        return k, v

    limit = args.driver_limit_gib * GIB if args.driver_limit_gib else 0
    peak_alloc = peak_driver = 0
    completed, done = True, 0
    StaticKVCache.update = instrumented_update
    try:
        with torch.inference_mode():
            model(torch.zeros(1, args.prompt, dtype=torch.long, device=device),
                  past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.mps.synchronize()
            start = time.perf_counter()
            for step in range(args.tokens):
                position = args.prompt + step
                model(
                    torch.zeros(1, 1, dtype=torch.long, device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1,
                    cache_position=torch.tensor([position], device=device),
                )
                done = step + 1
                if step % args.sample_every == 0 or step == args.tokens - 1:
                    torch.mps.synchronize()
                    peak_alloc = max(peak_alloc, torch.mps.current_allocated_memory())
                    peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
                    if limit and peak_driver > limit:
                        completed = False
                        break
            torch.mps.synchronize()
        seconds = time.perf_counter() - start
    finally:
        StaticKVCache.update = original_update

    return {
        "mode": args.mode,
        "requested_tokens": args.tokens,
        "completed_tokens": done,
        "completed": completed,
        "distinct_decode_kv_lengths": len(seen_lengths),
        "max_decode_kv_length": max(seen_lengths) if seen_lengths else None,
        "peak_allocated_gib": round(peak_alloc / GIB, 2),
        "peak_driver_gib": round(peak_driver / GIB, 2),
        "seconds": round(seconds, 2),
        "tokens_per_second": round(done / seconds, 2) if seconds else None,
        "bucket": args.bucket,
        "dtype": str(args.dtype).replace("torch.", ""),
    }


def child_result(args, mode) -> dict:
    command = [sys.executable, __file__, "--mode", mode, *args.forwarded]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-4:]
        return {"mode": mode, "completed": False, "error": " ".join(tail)[-300:]}
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["grow", "bucket", "both"], default="both")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--prompt", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=2048)
    parser.add_argument("--bucket", type=int, default=2048)
    parser.add_argument("--sample-every", type=int, default=32)
    parser.add_argument("--driver-limit-gib", type=float, default=16.0,
                        help="abort a policy once Metal driver memory exceeds this (0 disables)")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print(json.dumps({"error": "MPS is not available on this machine"}))
        return 2

    if args.mode in {"grow", "bucket"}:
        args.dtype = getattr(torch, args.dtype)
        print(json.dumps(run_mode(args)))
        return 0

    forwarded = ["--tokens", str(args.tokens), "--prompt", str(args.prompt),
                 "--layers", str(args.layers), "--hidden", str(args.hidden),
                 "--heads", str(args.heads), "--kv-heads", str(args.kv_heads),
                 "--head-dim", str(args.head_dim), "--vocab", str(args.vocab),
                 "--bucket", str(args.bucket), "--sample-every", str(args.sample_every),
                 "--driver-limit-gib", str(args.driver_limit_gib), "--dtype", args.dtype]
    args.forwarded = forwarded

    results = [child_result(args, mode) for mode in ("grow", "bucket")]
    print(json.dumps(results, indent=2))
    print()
    print("| decode policy | tokens | distinct K/V lengths | peak allocated GiB | "
          "peak driver GiB | tokens/s | completed |")
    print("|---|---|---|---|---|---|---|")
    for item in results:
        if "error" in item:
            print(f"| {item['mode']} | - | - | - | - | - | failed: {item['error'][:60]} |")
            continue
        print(f"| {item['mode']} | {item['completed_tokens']}/{item['requested_tokens']} | "
              f"{item['distinct_decode_kv_lengths']} | {item['peak_allocated_gib']} | "
              f"{item['peak_driver_gib']} | {item['tokens_per_second']} | "
              f"{'yes' if item['completed'] else 'no'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
