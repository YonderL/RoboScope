"""Pi-0 samples from the extracted HuggingFaceVLA LIBERO Spatial release.

Images stay in the published 256x256 RGB orientation. Episode tails repeat the
last real action, matching the raw-LIBERO Pi-0 loader.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_rgb(payload, size):
    from io import BytesIO

    from PIL import Image

    raw = payload["bytes"] if isinstance(payload, dict) else payload
    if not raw:
        raise ValueError("Missing embedded image bytes")
    image = Image.open(BytesIO(raw)).convert("RGB")
    array = np.asarray(image)
    if array.shape != (size, size, 3) or array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 {size}x{size} RGB, got {array.shape} {array.dtype}")
    return array


def build_lerobot_cache(root, cache, image_size=256):
    """Write one mmap per camera plus state/action arrays. Idempotent if complete."""
    import pyarrow.parquet as pq

    root, cache = Path(root), Path(cache)
    marker = cache / "complete.json"
    if marker.is_file() and json.loads(marker.read_text())["dataset_root"] == str(root.resolve()):
        return json.loads((cache / "index.json").read_text())
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    cache.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(cache / "state.npy", mode="w+", dtype=np.float32, shape=(total, 8))
    actions = np.lib.format.open_memmap(cache / "action.npy", mode="w+", dtype=np.float32, shape=(total, 7))
    cameras = {
        name: np.lib.format.open_memmap(
            cache / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(total, image_size, image_size, 3)
        )
        for name in ("agentview_rgb", "eye_in_hand_rgb")
    }
    episodes, cursor = [], 0
    for path in files:
        table = pq.ParquetFile(path).read().to_pydict()
        count = len(table["index"])
        for row in range(count):
            index = int(table["index"][row])
            if index != cursor:
                raise ValueError(f"Noncontiguous frame index in {path}: expected {cursor}, got {index}")
            state = np.asarray(table["observation.state"][row], dtype=np.float32)
            action = np.asarray(table["action"][row], dtype=np.float32)
            if state.shape != (8,) or action.shape != (7,):
                raise ValueError(f"Unexpected state/action shape at frame {index}")
            states[cursor] = state
            actions[cursor] = action
            cameras["agentview_rgb"][cursor] = _decode_rgb(table["observation.images.image"][row], image_size)
            cameras["eye_in_hand_rgb"][cursor] = _decode_rgb(
                table["observation.images.image2"][row], image_size
            )
            episode = int(table["episode_index"][row])
            task = int(table["task_index"][row])
            if not episodes or episodes[-1]["episode"] != episode:
                if episodes and episode != episodes[-1]["episode"] + 1:
                    raise ValueError("Episode indices are not contiguous")
                episodes.append({"episode": episode, "task": task, "start": cursor, "length": 1})
            else:
                if task != episodes[-1]["task"]:
                    raise ValueError(f"Task changed inside episode {episode}")
                episodes[-1]["length"] += 1
            cursor += 1
        del table
    if cursor != total:
        raise ValueError("Frame count changed while caching")
    for array in (states, actions, *cameras.values()):
        array.flush()
    index = {"frames": total, "image_size": image_size, "episodes": episodes}
    (cache / "index.json").write_text(json.dumps(index))
    marker.write_text(json.dumps({"dataset_root": str(root.resolve()), "frames": total}))
    return index


def prepare_lerobot_manifest(root, cache, languages, validation_fraction, split_seed):
    index = build_lerobot_cache(root, cache)
    by_task = {}
    for episode in index["episodes"]:
        by_task.setdefault(episode["task"], []).append(episode)
    if sorted(by_task) != list(range(len(languages))):
        raise ValueError("Task indices do not match the Spatial language list")
    rng = np.random.default_rng(split_seed)
    train, val = [], []
    for task, episodes in sorted(by_task.items()):
        order = rng.permutation(len(episodes))
        holdout = max(1, int(round(len(episodes) * validation_fraction)))
        if holdout >= len(episodes):
            raise ValueError(f"Task {task} has no training episodes after the holdout")
        chosen = {episodes[i]["episode"] for i in order[:holdout]}
        for episode in episodes:
            (val if episode["episode"] in chosen else train).append(episode)
    states = np.load(cache / "state.npy", mmap_mode="r")
    actions = np.load(cache / "action.npy", mmap_mode="r")
    stats = {}
    for name, values in (("state", states), ("action", actions)):
        parts = [values[episode["start"] : episode["start"] + episode["length"]] for episode in train]
        stacked = np.concatenate(parts).astype(np.float64)
        stats[f"pi0_{name}_mean"] = stacked.mean(0).tolist()
        stats[f"pi0_{name}_std"] = np.maximum(stacked.std(0), 1e-3).tolist()
    return {
        "dataset_format": "lerobot_v3",
        "dataset_root": str(Path(root).resolve()),
        "cache_root": str(Path(cache).resolve()),
        "pi0_adapter_version": 1,
        "pi0_state_keys": ["observation.state"],
        "pi0_action_semantics": "raw_7d_osc_delta_pose_and_gripper_no_extra_delta_transform",
        "pi0_padding": "repeat_last_action_supervised",
        "image_size": index["image_size"],
        "tasks": [{"id": i, "language": text} for i, text in enumerate(languages)],
        "train_episodes": train,
        "val_episodes": val,
        **stats,
    }


class LeRobotSpatialDataset(Dataset):
    def __init__(self, manifest, chunk_size, split):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.manifest = manifest
        self.chunk = chunk_size
        self.cache = Path(manifest["cache_root"])
        self.episodes = manifest[f"{split}_episodes"]
        self.ends = np.cumsum([episode["length"] for episode in self.episodes])
        self._arrays = None
        languages = {task["id"]: task["language"] for task in manifest["tasks"]}
        self.languages = [languages[episode["task"]] for episode in self.episodes]

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def _open(self):
        if self._arrays is None:
            self._arrays = {
                "state": np.load(self.cache / "state.npy", mmap_mode="r"),
                "action": np.load(self.cache / "action.npy", mmap_mode="r"),
                "agentview_rgb": np.load(self.cache / "agentview_rgb.npy", mmap_mode="r"),
                "eye_in_hand_rgb": np.load(self.cache / "eye_in_hand_rgb.npy", mmap_mode="r"),
            }
        return self._arrays

    def __getitem__(self, index):
        episode_id = int(np.searchsorted(self.ends, index, side="right"))
        start = int(self.ends[episode_id - 1]) if episode_id else 0
        offset = index - start
        episode = self.episodes[episode_id]
        frame = episode["start"] + offset
        arrays = self._open()
        remaining = episode["length"] - offset
        count = min(self.chunk, remaining)
        action = np.zeros((self.chunk, 7), dtype=np.float32)
        action[:count] = arrays["action"][frame : frame + count]
        action[count:] = action[count - 1]
        result = {
            "state": torch.from_numpy(np.array(arrays["state"][frame], copy=True)),
            "task_id": torch.tensor(episode["task"]),
            "action": torch.from_numpy(action),
            "action_is_pad": torch.arange(self.chunk) >= count,
            "task": self.languages[episode_id],
        }
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            result[key] = torch.from_numpy(np.array(arrays[key][frame], copy=True)).permute(2, 0, 1)
        return result

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}
