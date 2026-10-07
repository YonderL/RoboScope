"""Source releases exclude local probes and reject unexpected model artifacts."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_release", Path(__file__).resolve().parents[1] / "scripts/build_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def fixture_tree(root):
    for name in release.FILES:
        (root / name).write_text("fixture")
    (root / "scripts").mkdir()


def test_local_probe_is_not_in_source_archive(tmp_path):
    fixture_tree(tmp_path)
    probe = tmp_path / "scripts/benchmark_pi0_microbatch.py"
    probe.write_text("local hardware probe")
    public = tmp_path / "scripts/export_results.py"
    public.write_text("public exporter")
    files = release.release_files(tmp_path)
    assert public in files and probe not in files


@pytest.mark.parametrize("name", ["weights.safetensors", "weights.pth", "data.h5"])
def test_accidental_training_artifact_rejects_release(tmp_path, name):
    fixture_tree(tmp_path)
    (tmp_path / "scripts" / name).write_bytes(b"fixture")
    with pytest.raises(ValueError, match="artifact"):
        release.release_files(tmp_path)


def test_broken_symlink_rejects_release(tmp_path):
    fixture_tree(tmp_path)
    (tmp_path / "scripts/missing").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlinks"):
        release.release_files(tmp_path)
