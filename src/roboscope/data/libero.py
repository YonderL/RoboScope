"""Raw LIBERO HDF5 adapter. One epoch visits each training frame once, for every K."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare(cfg):
    """创建固定实验清单：任务 ID、按整条轨迹划分的训练/验证集和归一化统计。

    任务 ID 按文件名排序，而不是依赖 benchmark 的任务排列；评测按名称对应。
    所有 K 和训练 seed 共用这一份清单。只扫描小型 state/action 数组，
    图像保留在 HDF5，避免把整个 suite 图像复制到每个进程内存中。
    """
    paths = sorted((Path(cfg["data_root"]) / "libero_spatial").glob("*_demo.hdf5"))
    if len(paths) != 10:
        raise ValueError(f"Expected 10 spatial tasks, found {len(paths)}")
    result = {
        "schema": 1,
        "tasks": [],
        "split_seed": cfg["split_seed"],
        "validation_fraction": cfg["validation_fraction"],
        "state_keys": ["joint_states", "gripper_states"],
        "canonical_image_convention": "opencv",
        "control_freq": 20,
    }
    vectors = {"state": [], "action": []}
    for task_id, path in enumerate(paths):
        name = path.name.removesuffix("_demo.hdf5")
        root = Path(cfg["libero_root"])
        bddl = root / "bddl_files/libero_spatial" / (name + ".bddl")
        init = root / "init_files/libero_spatial" / (name + ".pruned_init")
        if not bddl.is_file() or not init.is_file():
            raise FileNotFoundError(f"Missing BDDL or initial states for {name}")
        initial_states = torch.load(init, map_location="cpu", weights_only=False)
        if len(initial_states) < cfg["eval_episodes"]:
            raise ValueError("Too few unique initial states; refusing to wrap indices")
        with h5py.File(path, "r") as f:
            group = f["data"]
            convention = group.attrs["macros_image_convention"]
            if isinstance(convention, bytes):
                convention = convention.decode()
            if convention not in ("opengl", "opencv"):
                raise ValueError(f"Unsupported image convention {convention}")
            env = json.loads(group.attrs["env_args"])["env_kwargs"]
            if env["controller_configs"]["type"] != "OSC_POSE" or env["control_freq"] != 20:
                raise ValueError("This adapter expects OSC_POSE at 20 Hz")
            names = sorted(group, key=lambda s: int(s.split("_")[-1]))
            shuffled = np.random.default_rng(cfg["split_seed"] + task_id).permutation(names)
            n_val = max(1, round(len(names) * cfg["validation_fraction"]))
            val = set(shuffled[:n_val])
            episodes = []
            for demo in names:
                d = group[demo]
                actions = np.asarray(d["actions"], dtype=np.float32)
                state = np.concatenate([d["obs/joint_states"][:], d["obs/gripper_states"][:]], -1).astype(
                    np.float32
                )
                if state.shape != (len(actions), 9) or actions.shape[1:] != (7,):
                    raise ValueError(f"Unexpected dimensions in {path}/{demo}")
                if not np.isfinite(state).all() or not np.isfinite(actions).all():
                    raise ValueError("Nonfinite data")
                if not len(actions):
                    raise ValueError("Empty demonstration")
                for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                    if d[f"obs/{camera}"].shape != (len(actions), 128, 128, 3):
                        raise ValueError("Expected 128x128 RGB observations")
                split = "val" if demo in val else "train"
                episodes.append({"demo": demo, "length": len(actions), "split": split})
                if split == "train":
                    vectors["state"].append(state)
                    vectors["action"].append(actions)
            stat = path.stat()
            result["tasks"].append(
                {
                    "id": task_id,
                    "name": name,
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "image_convention": convention,
                    "episodes": episodes,
                    "bddl": str(bddl),
                    "bddl_sha256": digest(bddl),
                    "init": str(init),
                    "init_sha256": digest(init),
                    "controller": env["controller_configs"],
                    "eval_initial_state_ids": list(range(cfg["eval_episodes"])),
                }
            )
    for key, items in vectors.items():
        arr = np.concatenate(items).astype(np.float64)
        result[key + "_mean"] = arr.mean(0).tolist()
        result[key + "_std"] = np.maximum(arr.std(0), 1e-3).tolist()
    return result


class FrameDataset(Dataset):
    def __init__(self, manifest, chunk, split, image_cache=None):
        self.tasks = manifest["tasks"]
        self.chunk = chunk
        self.episodes = []
        self.ends = []
        self.files = OrderedDict()
        self.image_cache = Path(image_cache) if image_cache else None
        self.cached_arrays = {}
        self.cache_index = json.loads((self.image_cache / "index.json").read_text()) if image_cache else None
        total = 0
        for task in self.tasks:
            with h5py.File(task["path"], "r") as f:
                for ep in task["episodes"]:
                    if ep["split"] != split:
                        continue
                    d = f["data"][ep["demo"]]
                    state = np.concatenate([d["obs/joint_states"][:], d["obs/gripper_states"][:]], -1).astype(
                        np.float32
                    )
                    actions = np.asarray(d["actions"], dtype=np.float32)
                    self.episodes.append((task["id"], ep["demo"], state, actions))
                    total += len(actions)
                    self.ends.append(total)
        self.ends = np.asarray(self.ends)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        """一个索引代表一条轨迹中的起始帧 t，与 K 无关。

        使用原始 LIBERO 文件的同索引配对：obs[t] -> actions[t:t+K]。
        末尾不足 K 的位置补零，并用 action_is_pad 排除损失和 VAE attention。
        不跨轨迹、不删掉短尾样本，因此 K 的改变不会改变 epoch 的样本数。
        注意：这是发布数据的索引约定，并未擅自做机器人平台的时间偏移。
        """
        episode = int(np.searchsorted(self.ends, index, side="right"))
        t = index - (int(self.ends[episode - 1]) if episode else 0)
        task_id, demo, states, actions = self.episodes[episode]
        task = self.tasks[task_id]
        if self.image_cache is None and task_id not in self.files:
            if len(self.files) >= 10:
                _, old = self.files.popitem(last=False)
                old.close()
            self.files[task_id] = h5py.File(task["path"], "r")
        if self.image_cache is None:
            self.files.move_to_end(task_id)
            obs = self.files[task_id]["data"][demo]["obs"]
        n = min(self.chunk, len(actions) - t)
        chunk = np.zeros((self.chunk, 7), dtype=np.float32)
        chunk[:n] = actions[t : t + n]
        result = {
            "state": torch.from_numpy(states[t].copy()),
            "task_id": torch.tensor(task_id),
            "action": torch.from_numpy(chunk),
            "action_is_pad": torch.arange(self.chunk) >= n,
        }
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            if self.image_cache is None:
                img = obs[key][t]
            else:
                cache_key = (task_id, key)
                if cache_key not in self.cached_arrays:
                    self.cached_arrays[cache_key] = np.load(
                        self.image_cache / f"task{task_id}_{key}.npy", mmap_mode="r"
                    )
                offset = self.cache_index["tasks"][str(task_id)]["offsets"][demo]
                img = self.cached_arrays[cache_key][offset + t]
            # 数据元信息为 opengl 时上下翻转；线上设置 robosuite 为 opencv。
            # 训练和 rollout 最终都给模型相同的 RGB / CHW / 图像方向。
            if task["image_convention"] == "opengl":
                img = img[::-1]
            result[key] = torch.from_numpy(img.copy()).permute(2, 0, 1)
        return result

    def __getstate__(self):
        return {**self.__dict__, "files": OrderedDict(), "cached_arrays": {}}
