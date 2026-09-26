"""Shared training state, batch transfer and atomic checkpoint utilities.

Keep these helpers independent of any policy or trainer implementation.
"""

import hashlib
import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


@contextmanager
def fixed_rng(seed, device):
    saved = capture_rng(device)
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # manual_seed also seeds CUDA generators; use the CPU generator directly
        # so validation never changes another device's RNG in this process.
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
    finally:
        restore_rng(saved, device)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_checkpoint(path, payload):
    atomic_write(path, lambda handle: torch.save(payload, handle))


def save_json(path, value):
    contents = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda handle: handle.write(contents))


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value
