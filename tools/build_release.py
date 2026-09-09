#!/usr/bin/env python3
"""Build the Python distribution and portable YuE2 skill with matching checksums.

Run after installing build requirements:
python -m pip install 'setuptools>=77' build wheel.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


def build(root: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(root)],
        check=True,
    )
    skill = root / "skills" / "yue2-music"
    members = []
    for path in sorted(skill.rglob("*")):
        relative = path.relative_to(skill)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks cannot be distributed: {relative}")
        if not path.is_file() or path.suffix == ".pyc":
            continue
        if relative.parts[0] not in {"SKILL.md", "LICENSE", "agents", "assets", "references", "scripts"}:
            raise ValueError(f"Unexpected skill file: {relative}")
        members.append((path, "yue2-music/" + relative.as_posix()))
    if not any(name == "yue2-music/SKILL.md" for _, name in members):
        raise ValueError("Missing skill entrypoint")
    archive = output / "yue2-music.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path, name in members:
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            package.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert set(package.namelist()) == {name for _, name in members}
        for path, name in members:
            assert package.read(name) == path.read_bytes()
    artifacts = [output / f"yue2_infer-{version}-py3-none-any.whl",
                 output / f"yue2_infer-{version}.tar.gz", archive]
    hashes = []
    for path in artifacts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.name}\n")
    (output / "SHA256SUMS").write_text("".join(hashes))
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    for artifact in build(Path(__file__).resolve().parents[1], args.output.resolve()):
        print(artifact.name)
