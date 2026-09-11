"""Lossless raw RGB cache shared by all runs. No resizing, augmentation or frame filtering."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from roboscope.data.libero import save_json


def prepare_cache(manifest, folder):
    """原始像素按任务拼成 .npy，通过 mmap 共享 OS page cache。

    各训练 worker 不再为每个随机帧查 HDF5 group/dataset。缓存仍是 uint8 原图，
    flip/normalization 保留在原 adapter；改变的仅是存储读取方式。
    未写完的缓存没有完成标记，下次启动会重建，不把部分文件当有效数据。
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        json.dumps(
            [{k: t[k] for k in ("path", "size", "mtime_ns", "episodes")} for t in manifest["tasks"]],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    index_path = folder / "index.json"
    if index_path.exists():
        old = json.loads(index_path.read_text())
        if old["signature"] != signature:
            raise RuntimeError("Image cache does not match dataset manifest")
        for key, entry in old["tasks"].items():
            for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                arr = np.load(folder / f"task{key}_{camera}.npy", mmap_mode="r")
                if arr.shape != (entry["frames"], 128, 128, 3) or arr.dtype != np.uint8:
                    raise ValueError("Invalid image cache")
        return
    index = {"signature": signature, "tasks": {}}
    for task in manifest["tasks"]:
        frames = sum(e["length"] for e in task["episodes"])
        offsets = {}
        with h5py.File(task["path"], "r") as f:
            for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                target = folder / f"task{task['id']}_{camera}.npy"
                temp = target.with_suffix(".tmp.npy")
                arr = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(frames, 128, 128, 3))
                start = 0
                for episode in task["episodes"]:
                    offsets[episode["demo"]] = start
                    stop = start + episode["length"]
                    arr[start:stop] = f[f"data/{episode['demo']}/obs/{camera}"][:]
                    start = stop
                arr.flush()
                del arr
                temp.replace(target)
        index["tasks"][str(task["id"])] = {"frames": frames, "offsets": offsets}
        print(f"Image cache task {task['id']}: {frames} frames", flush=True)
    save_json(index_path, index)
