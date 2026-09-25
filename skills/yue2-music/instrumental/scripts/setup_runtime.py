#!/usr/bin/env python3
"""Install the pinned public YuE2 runtime, download models, or verify a GPU host."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

from common import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def check_model(path, spec):
    checks = {}
    for name, expected in spec['core_files'].items():
        target = path / name
        if not target.is_file() or target.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing/wrong-sized model file: {target}')
        digest = hashlib.sha256() if 'sha256' in expected else hashlib.sha1()
        if 'git_blob_sha1' in expected:
            digest.update(f"blob {expected['bytes']}\0".encode())
        with target.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected.get('sha256', expected.get('git_blob_sha1')):
            raise ValueError(f'Model identity mismatch: {target}')
        checks[name] = digest.hexdigest()
    return checks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    install = commands.add_parser('install', help='Create a fresh venv and install the pinned official wheel and dependencies')
    install.add_argument('--venv', type=Path, required=True)
    download = commands.add_parser('models', help='Download only runtime model assets at frozen revisions')
    download.add_argument('--output', type=Path, required=True)
    download.add_argument('--offline', action='store_true', help='Resolve only files already in the Hugging Face cache')
    doctor = commands.add_parser('doctor', help='Check runtime versions, BF16 GPU, and optional model identities')
    doctor.add_argument('--models-root', type=Path)
    doctor.add_argument('--require-gpu', action='store_true')
    doctor.add_argument('--device', default='cuda:0')
    doctor.add_argument('--output', type=Path)
    args = cli.parse_args()
    release = read_json(ROOT / 'assets/release.json')
    try:
        if args.command == 'install':
            if sys.version_info < (3, 10):
                raise RuntimeError('Use Python 3.10+; Python 3.12 is the tested version')
            if args.venv.exists():
                raise ValueError('Choose a new environment path; existing environments are not modified')
            venv.EnvBuilder(with_pip=True).create(args.venv)
            python = args.venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-inference.txt')], check=True)
            subprocess.run([str(python), str(Path(__file__).resolve()), 'doctor'], check=True)
            print(json.dumps({'python': str(python), 'note': 'Run models and generate with this interpreter; GPU doctor runs inside the GPU allocation.'}))
            return 0
        if args.command == 'models':
            from huggingface_hub import snapshot_download
            args.output.mkdir(parents=True, exist_ok=True)
            verified = {}
            for name, spec in release['models'].items():
                target = args.output / name
                snapshot_download(repo_id=spec['repo'], revision=spec['revision'], local_dir=target,
                    local_files_only=args.offline,
                    allow_patterns=list(spec['core_files']) + ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'licenses/*'])
                verified[name] = check_model(target, spec)
            write_json(args.output / 'models-verified.json', {'passed': True, 'models': release['models'], 'files': verified})
            print(json.dumps({'models_root': str(args.output), 'passed': True}))
            return 0
        expected = {'yue2-infer': release['package_version'], 'torch': '2.10.0', 'transformers': '4.57.6',
                    'huggingface-hub': '0.36.2', 'numpy': '2.2.6', 'soundfile': '0.13.1',
                    'tiktoken': '0.12.0', 'safetensors': '0.7.0', 'accelerate': '1.13.0'}
        versions = {name: importlib.metadata.version(name) for name in expected}
        bad = {name: value for name, value in versions.items() if value.split('+')[0] != expected[name]}
        if bad:
            raise ValueError(f'Runtime versions differ from the recipe: {bad}')
        import torch
        cuda = torch.cuda.is_available()
        ready = False
        gpu = None
        if cuda:
            device = torch.device(args.device)
            props = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            ready = props.major >= 8 and total >= 22 * 2**30 and free >= 20 * 2**30
            gpu = {'name': props.name, 'total_gib': total / 2**30, 'free_gib': free / 2**30, 'bf16': props.major >= 8}
        if args.require_gpu and not ready:
            raise RuntimeError('Need a BF16 GPU in the supported 24 GB class with sufficient free memory; run inside a suitable allocation')
        model_checks = None
        if args.models_root:
            model_checks = {name: check_model(args.models_root / name, spec) for name, spec in release['models'].items()}
        result = {'passed': True, 'versions': versions, 'gpu_ready': ready, 'gpu': gpu,
                  'models': model_checks, 'quality_validated': False}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, importlib.metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
