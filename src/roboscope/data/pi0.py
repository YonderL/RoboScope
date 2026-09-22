"""Versioned Pi-0 inputs on the existing raw LIBERO trajectory split.

State is end-effector xyz + axis-angle + two gripper positions (8D). Actions
remain the dataset's 7D OSC_POSE commands: they are already relative commands.
"""

import json

import h5py
import numpy as np
import torch

from roboscope.data.libero import FrameDataset, prepare


def prepare_pi0(cfg):
    manifest = prepare(cfg)
    manifest.update(
        pi0_adapter_version=1,
        pi0_state_keys=["ee_states", "gripper_states"],
        pi0_action_semantics="raw_7d_osc_delta_pose_and_gripper_no_extra_delta_transform",
        pi0_padding="repeat_last_action_supervised",
        language_source="data.attrs.problem_info.language_instruction",
    )
    states, actions = [], []
    for task in manifest["tasks"]:
        with h5py.File(task["path"], "r") as handle:
            group = handle["data"]
            language = json.loads(group.attrs["problem_info"])["language_instruction"]
            if not isinstance(language, str) or not language.strip():
                raise ValueError(f"Missing natural-language instruction for {task['name']}")
            task["language"] = language.strip()
            for episode in task["episodes"]:
                demo = group[episode["demo"]]
                state = read_state(demo["obs"])
                if state.shape != (episode["length"], 8) or not np.isfinite(state).all():
                    raise ValueError(f"Invalid end-effector state in {task['name']}/{episode['demo']}")
                if episode["split"] == "train":
                    states.append(state)
                    actions.append(np.asarray(demo["actions"], dtype=np.float32))
    for name, items in (("state", states), ("action", actions)):
        values = np.concatenate(items).astype(np.float64)
        manifest[f"pi0_{name}_mean"] = values.mean(0).tolist()
        manifest[f"pi0_{name}_std"] = np.maximum(values.std(0), 1e-3).tolist()
    return manifest


def read_state(obs):
    return np.concatenate([obs["ee_states"][:], obs["gripper_states"][:]], axis=-1).astype(np.float32)


class Pi0Dataset(FrameDataset):
    """Reuse frame indexing, camera convention and optional mmap image cache."""

    def __init__(self, manifest, chunk_size, split, image_cache=None):
        super().__init__(manifest, chunk_size, split, image_cache)
        replaced = []
        handles = {}
        try:
            for task_id, demo, _, actions in self.episodes:
                if task_id not in handles:
                    handles[task_id] = h5py.File(self.tasks[task_id]["path"], "r")
                state = read_state(handles[task_id][f"data/{demo}/obs"])
                replaced.append((task_id, demo, state, actions))
        finally:
            for handle in handles.values():
                handle.close()
        self.episodes = replaced

    def __getitem__(self, index):
        result = super().__getitem__(index)
        # Pi-0's fixed horizon supervises a repeated terminal action, as in
        # the reference sequence loader; never crosses an episode boundary.
        valid = int((~result["action_is_pad"]).sum())
        result["action"][valid:] = result["action"][valid - 1]
        result["task"] = self.tasks[int(result["task_id"])]["language"]
        return result


def quaternion_to_axis_angle(quaternion):
    """LIBERO / robosuite xyzw quaternion convention, matching saved ee_ori."""
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Expected a finite xyzw quaternion")
    # Match robosuite.utils.transform_utils.quat2axisangle, including the
    # quaternion sign convention rather than applying a hemisphere change.
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] ** 2)
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / denominator)).astype(np.float32)


def observation_batch(obs, task, device):
    """Raw OpenCV-convention simulator observation -> batch of one."""
    state = np.concatenate(
        [obs["robot0_eef_pos"], quaternion_to_axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Invalid Pi-0 rollout state")
    language = task["language"] if isinstance(task, dict) else task
    result = {"state": torch.from_numpy(state).unsqueeze(0).to(device), "task": [language]}
    for name, key in (("agentview_rgb", "agentview_image"), ("eye_in_hand_rgb", "robot0_eye_in_hand_image")):
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected uint8 HWC RGB simulator cameras")
        result[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    return result
