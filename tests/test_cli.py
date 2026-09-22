import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_training_preview_needs_neither_data_nor_cuda(tmp_path):
    output = tmp_path / "never-created"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roboscope",
            "train",
            "--recipe",
            str(ROOT / "configs/libero_spatial/diffusion.json"),
            "--data-root",
            "/missing/data",
            "--libero-root",
            "/missing/libero",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Preview only" in result.stdout
    assert not output.exists()


def test_evaluation_preview_does_not_load_weights(tmp_path):
    output = tmp_path / "no-evaluation"
    result = subprocess.run(
        [sys.executable, "-m", "roboscope", "evaluate", "--source", "/missing/run", "--output", str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "No checkpoint loaded" in result.stdout
    assert not output.exists()


def test_worker_entrypoints_import_when_training_stack_installed():
    import importlib.util

    import pytest

    if importlib.util.find_spec("lerobot") is None:
        pytest.skip("Optional training dependencies not installed")
    for module in [
        "roboscope.trainers.act",
        "roboscope.trainers.diffusion",
        "roboscope.trainers.smolvla",
        "roboscope.trainers.smolvla_rlt",
        "roboscope.evaluation.worker",
    ]:
        subprocess.run([sys.executable, "-m", module, "--help"], check=True, capture_output=True)
