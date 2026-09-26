"""Decoded uint8 memmap for the official Spatial images.

LeRobot otherwise decodes a PNG from parquet on every sample. The cache stores
the same RGB bytes, in dataset order, and the training process serves them as
the float CHW tensors ``ToTensor`` would have produced.
"""

import json
import os
from pathlib import Path

import numpy as np

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
NUMERIC_KEYS = ("observation.state", "action")


def cache_dir_for(dataset):
    return Path(dataset).resolve().parent / (Path(dataset).name + "_image_cache")


def _signature(dataset):
    dataset = Path(dataset)
    provenance = json.loads((dataset / "spatial_provenance.json").read_text())
    info = json.loads((dataset / "meta/info.json").read_text())
    return {
        "source_revision": provenance["source_revision"],
        "episodes": provenance["source_episode_indices"],
        "frames": info["total_frames"],
        "shape": [256, 256, 3],
    }


def _decode_file(task):
    import pyarrow.parquet as pq
    from PIL import Image

    path, keys = task
    table = pq.read_table(path, columns=["index", *keys])
    frames = {key: [] for key in keys}
    indices = table.column("index").to_pylist()
    for key in keys:
        column = table.column(key).to_pylist()
        decoded = []
        for value in column:
            image = Image.open(__import__("io").BytesIO(value["bytes"])).convert("RGB")
            decoded.append(np.asarray(image, dtype=np.uint8))
        frames[key] = decoded
    return indices, frames


def build_image_cache(dataset, folder=None, workers=8):
    """Write one read-only memmap per camera. A finished index.json marks success."""
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow.parquet as pq

    dataset = Path(dataset).resolve()
    folder = Path(folder) if folder else cache_dir_for(dataset)
    folder.mkdir(parents=True, exist_ok=True)
    signature = _signature(dataset)
    receipt = folder / "index.json"
    if receipt.exists() and json.loads(receipt.read_text())["signature"] == signature:
        return folder
    if any(folder.iterdir()):
        raise RuntimeError(f"Incomplete image cache: {folder}")
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
    if sum(counts) != signature["frames"]:
        raise RuntimeError("Parquet frame count does not match info.json")
    maps = {
        key: np.lib.format.open_memmap(
            folder / f"{key}.npy", mode="w+", dtype=np.uint8, shape=(signature["frames"], 256, 256, 3)
        )
        for key in IMAGE_KEYS
    }
    tasks = [(path, IMAGE_KEYS) for path in files]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for indices, frames in pool.map(_decode_file, tasks, chunksize=1):
            for key in IMAGE_KEYS:
                maps[key][indices] = np.stack(frames[key])
            done += len(indices)
            print(f"Decoded {done}/{signature['frames']} frames", flush=True)
    for array in maps.values():
        array.flush()
    del maps
    receipt.write_text(json.dumps({"signature": signature, "keys": list(IMAGE_KEYS)}, indent=2) + "\n")
    print(f"Image cache ready: {folder}", flush=True)
    return folder


def _as_model_tensor(frame):
    """Match torchvision ToTensor: uint8 HWC RGB to float32 CHW in [0, 1]."""
    import torch

    copied = np.array(frame, dtype=np.uint8, copy=True)
    return torch.from_numpy(copied).permute(2, 0, 1).contiguous().float().div_(255)


def build_numeric_cache(dataset, folder=None):
    """Memmap state and action so a 50-step window is a slice, not a parquet gather."""
    import pyarrow.parquet as pq

    dataset = Path(dataset).resolve()
    folder = Path(folder) if folder else cache_dir_for(dataset)
    folder.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset / "meta/info.json").read_text())
    frames = int(info["total_frames"])
    if all((folder / f"{key}.npy").is_file() for key in NUMERIC_KEYS):
        return folder
    arrays = {}
    for key in NUMERIC_KEYS:
        width = int(info["features"][key]["shape"][0])
        arrays[key] = np.lib.format.open_memmap(
            folder / f"{key}.npy", mode="w+", dtype=np.float32, shape=(frames, width)
        )
    done = 0
    for path in sorted((dataset / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["index", *NUMERIC_KEYS])
        indices = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
        for key in NUMERIC_KEYS:
            arrays[key][indices] = np.asarray(table.column(key).to_pylist(), dtype=np.float32)
        done += len(indices)
        print(f"Cached numeric columns {done}/{frames}", flush=True)
    for array in arrays.values():
        array.flush()
    return folder


def _load_maps(folder):
    folder = Path(folder)
    meta = json.loads((folder / "index.json").read_text())
    maps = {key: np.load(folder / f"{key}.npy", mmap_mode="r") for key in meta["keys"]}
    for key in NUMERIC_KEYS:
        path = folder / f"{key}.npy"
        if path.is_file():
            maps[key] = np.load(path, mmap_mode="r")
    return maps


def install(folder):
    """Replace PNG decoding inside this process. DataLoader workers import it again."""
    import torch
    from lerobot.datasets.dataset_reader import DatasetReader

    if getattr(DatasetReader.get_item, "_roboscope_image_cache", False):
        return
    maps = _load_maps(folder)
    image_maps = {key: maps[key] for key in IMAGE_KEYS}
    numeric_maps = {key: maps[key] for key in NUMERIC_KEYS if key in maps}
    original_query = DatasetReader._query_hf_dataset

    def row(dataset, idx):
        table = dataset.data
        item = {}
        for key in dataset.column_names:
            if key in maps:
                continue
            value = table.column(key)[idx].as_py()
            item[key] = value if isinstance(value, str) else torch.tensor(value)
        position = int(item["index"])
        for key, array in image_maps.items():
            item[key] = _as_model_tensor(array[position])
        for key, array in numeric_maps.items():
            item[key] = torch.from_numpy(np.array(array[position], dtype=np.float32, copy=True))
        return item

    def get_item(self, idx):
        item = row(self.hf_dataset, idx)
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            item = {**item, **padding, **self._query_hf_dataset(query_indices)}
        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            item = {**self._query_videos(query_timestamps, ep_idx), **item}
        if self._image_transforms is not None:
            for cam in self._meta.camera_keys:
                if cam in self._meta.depth_keys:
                    continue
                item[cam] = self._image_transforms(item[cam])
        item["task"] = self._meta.tasks.iloc[item["task_index"].item()].name
        return item

    def query(self, query_indices):
        result = {}
        cached = {
            key: values for key, values in query_indices.items() if key in image_maps or key in numeric_maps
        }
        remainder = {key: values for key, values in query_indices.items() if key not in cached}
        if remainder:
            result.update(original_query(self, remainder))
        for key, indices in cached.items():
            if key in numeric_maps:
                result[key] = torch.from_numpy(
                    np.array(numeric_maps[key][indices], dtype=np.float32, copy=True)
                )
                continue
            frames = np.stack([np.array(image_maps[key][i], copy=True) for i in indices])
            result[key] = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().div_(255)
        return result

    get_item._roboscope_image_cache = True
    DatasetReader.get_item = get_item
    DatasetReader._query_hf_dataset = query
    os.environ["ROBOSCOPE_IMAGE_CACHE"] = str(folder)
