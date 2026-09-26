"""Build an explicit source allowlist; no datasets, checkpoints or legacy experiments."""

import argparse
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "README.md",
    "README.zh-CN.md",
    "pyproject.toml",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    ".gitignore",
    "requirements-training.txt",
    "requirements-smolvla.txt",
    "requirements-pi0.txt",
    "constraints-lerobot.txt",
]
DIRS = ["src", "configs", "docs", "examples", "tests", "scripts", "results", ".github"]
ARTIFACT_SUFFIXES = {".pt", ".pth", ".safetensors", ".hdf5", ".h5", ".npz", ".npy", ".mp4", ".pyc"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("dist/roboscope-source.tar.gz"))
    a = p.parse_args()
    paths = release_files(ROOT)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=Path("roboscope") / path.relative_to(ROOT), recursive=False)
    print(f"{len(paths)} files; {a.output.stat().st_size / 1024 / 1024:.2f} MiB; {a.output}")


def release_files(root):
    paths = [root / name for name in FILES]
    for directory in DIRS:
        for path in (root / directory).rglob("*"):
            if any(x in path.parts for x in ("__pycache__", ".pytest_cache", ".ruff_cache")) or any(
                x.endswith(".egg-info") for x in path.parts
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"Release refuses symlinks: {path}")
            # PNG previews are sufficient for Git; vector/PDF exports are regenerated.
            if path.parent == root / "docs/assets" and path.suffix in (".pdf", ".svg"):
                continue
            if path.is_file():
                paths.append(path)
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Release refuses symlinks: {path}")
        if path.suffix in ARTIFACT_SUFFIXES:
            raise ValueError(f"Unexpected artifact: {path}")
        if path.suffix in (".py", ".md", ".json", ".toml", ".txt", ".csv", ".yml", ".yaml", ".sh"):
            content = path.read_text()
            if re.search(r"/(?:home|data)/[A-Za-z0-9_-]+/", content):
                raise ValueError(f"Nonportable personal path in release: {path}")
    return sorted(set(paths))


if __name__ == "__main__":
    main()
