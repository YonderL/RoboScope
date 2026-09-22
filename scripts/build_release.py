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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("dist/roboscope-source.tar.gz"))
    a = p.parse_args()
    paths = [ROOT / name for name in FILES]
    for directory in DIRS:
        paths += [
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and not any(x in p.parts for x in ("__pycache__", ".pytest_cache", ".ruff_cache"))
            and not any(x.endswith(".egg-info") for x in p.parts)
        ]
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Release refuses symlinks: {path}")
        if path.suffix in (".pt", ".hdf5", ".npz", ".npy", ".mp4", ".pyc"):
            raise ValueError(f"Unexpected artifact: {path}")
        if path.suffix in (".py", ".md", ".json", ".toml", ".txt", ".csv", ".yml"):
            content = path.read_text()
            if re.search(r"/(?:home|data)/[A-Za-z0-9_-]+/", content):
                raise ValueError(f"Nonportable personal path in release: {path}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, "w:gz") as archive:
        for path in sorted(set(paths)):
            archive.add(path, arcname=Path("roboscope") / path.relative_to(ROOT), recursive=False)
    print(f"{len(set(paths))} files; {a.output.stat().st_size / 1024 / 1024:.2f} MiB; {a.output}")


if __name__ == "__main__":
    main()
