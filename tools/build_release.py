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


def build_skill(root: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    skill = root / "skills" / "yue2-music"
    subprocess.run(
        [sys.executable, "-B", str(skill / "instrumental/scripts/verify_bundle.py"), "--strict"],
        check=True,
    )
    members = []
    for path in sorted(skill.rglob("*")):
        relative = path.relative_to(skill)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks cannot be distributed: {relative}")
        if not path.is_file() or path.suffix == ".pyc":
            continue
        if relative.parts[0] not in {"SKILL.md", "LICENSE", "agents", "assets", "references", "scripts", "instrumental"}:
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
    return archive


def write_checksums(output: Path, artifacts: list[Path]) -> None:
    hashes = []
    for path in artifacts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.name}\n")
    (output / "SHA256SUMS").write_text("".join(hashes))


def build(root: Path, output: Path) -> list[Path]:
    archive = build_skill(root, output)
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(root)],
        check=True,
    )
    artifacts = [output / f"yue2_infer-{version}-py3-none-any.whl",
                 output / f"yue2_infer-{version}.tar.gz", archive]
    write_checksums(output, artifacts)
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--skill-only", action="store_true", help="Package the skill without rebuilding the Python runtime")
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    if args.skill_only:
        artifacts = [build_skill(root, output)]
        write_checksums(output, artifacts)
    else:
        artifacts = build(root, output)
    for artifact in artifacts:
        print(artifact.name)
