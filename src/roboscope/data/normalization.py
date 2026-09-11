"""Train-only normalization statistics. Never fit on evaluation trajectories."""

import h5py
import numpy as np


def add_action_limits(manifest):
    low, high = np.full(7, np.inf), np.full(7, -np.inf)
    for task in manifest["tasks"]:
        with h5py.File(task["path"], "r") as handle:
            for episode in task["episodes"]:
                if episode["split"] == "train":
                    actions = np.asarray(handle["data"][episode["demo"]]["actions"], dtype=np.float32)
                    low = np.minimum(low, actions.min(0))
                    high = np.maximum(high, actions.max(0))
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("No finite training action range")
    manifest.update(action_min=low.tolist(), action_max=high.tolist())
