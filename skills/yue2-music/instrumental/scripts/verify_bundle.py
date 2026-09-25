#!/usr/bin/env python3
"""Verify shipped skill files against their manifest; requires only Python."""
import argparse
import json
from pathlib import Path
import sys

from common import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify(root, strict=False):
    manifest = json.loads((root / 'bundle-manifest.json').read_text(encoding='utf-8'))
    required = {'SKILL.md', 'README.md', 'PRIVACY.md', 'assets/release.json', 'requirements-inference.txt',
        'scripts/instrumental.py', 'scripts/abc_tools.py', 'scripts/compile_score.py',
        'scripts/common.py', 'scripts/setup_runtime.py', 'scripts/verify_bundle.py', 'scripts/test_tools.py'}
    required.update({'scripts/instrumentalize.py', 'scripts/transcribe_cover.py', 'assets/cover-release.json'})
    if not required <= set(manifest['files']):
        raise ValueError('Manifest is incomplete')
    if strict:
        expected = set(manifest['files']) | {'bundle-manifest.json'}
        actual = set()
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Distribution must not contain symbolic links')
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        if actual != expected:
            raise ValueError('Distribution contains missing or unlisted files')
    for name, expected in manifest['files'].items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe manifest path')
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Missing/corrupt bundled file: {name}')
    return len(manifest['files'])


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--strict', action='store_true', help='For clean release folders: reject extra files and symlinks')
    args = cli.parse_args()
    try:
        print(json.dumps({'passed': True, 'files': verify(ROOT, strict=args.strict), 'scope': 'Bundled files only; install models and generate a fresh sample on the target GPU'}))
    except (OSError, ValueError, KeyError) as exc:
        print(f'Bundle verification failed: {exc}', file=sys.stderr)
        sys.exit(2)
