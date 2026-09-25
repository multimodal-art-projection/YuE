"""Check that the downloadable skill includes its fixtures and excludes local data."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("skill_release", ROOT / "tools/build_release.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class SkillReleaseTests(unittest.TestCase):
    def test_reproducible_archive_contains_complete_instrumental_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            first = BUILD.build_skill(ROOT, Path(directory) / "first")
            second = BUILD.build_skill(ROOT, Path(directory) / "second")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = set(archive.namelist())
                for relative in ("SKILL.md", "scripts/run_yue2.py", "instrumental/SKILL.md",
                                 "instrumental/scripts/instrumental.py", "instrumental/scripts/transcribe_cover.py",
                                 "instrumental/assets/example-ins.abc", "instrumental/assets/example-style.txt",
                                 "instrumental/requirements-inference.txt", "instrumental/bundle-manifest.json"):
                    self.assertIn("yue2-music/" + relative, names)
                for member in archive.infolist():
                    self.assertFalse(member.extra or member.comment)
                    self.assertFalse(member.filename.endswith((".mp3", ".flac", ".npy", ".log", ".pyc")))

    def fixture(self, root):
        skill = root / "skills/yue2-music"
        shutil.copytree(ROOT / "skills/yue2-music", skill)
        return skill

    def test_rejects_unlisted_private_work_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = self.fixture(root)
            (skill / "instrumental/.private-run.log").write_text("Synthetic private work file")
            with self.assertRaises(subprocess.CalledProcessError):
                BUILD.build_skill(root, root / "dist")
            self.assertFalse((root / "dist/yue2-music.zip").exists())

    def test_rejects_changed_manifest_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = self.fixture(root)
            (skill / "instrumental/assets/example-style.txt").write_text("Changed without updating manifest")
            with self.assertRaises(subprocess.CalledProcessError):
                BUILD.build_skill(root, root / "dist")
            self.assertFalse((root / "dist/yue2-music.zip").exists())


if __name__ == "__main__":
    unittest.main()
