"""Language-conditioned SmolVLA on the ACT/DP trajectory split and observations."""

import json

import h5py

from roboscope.data.libero import FrameDataset, prepare


def prepare_smolvla(cfg):
    manifest = prepare(cfg)
    manifest.update(smolvla_adapter_version=1, language_source="data.attrs.problem_info.language_instruction")
    for task in manifest["tasks"]:
        with h5py.File(task["path"], "r") as handle:
            language = json.loads(handle["data"].attrs["problem_info"])["language_instruction"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"Missing language instruction for {task['name']}")
        task["language"] = language.strip()
    return manifest


class SmolVLADataset(FrameDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        # LeRobot clamps queries past the episode end to the last frame/action,
        # and supplies action_is_pad to the native SmolVLA loss.
        valid = int((~result["action_is_pad"]).sum())
        result["action"][valid:] = result["action"][valid - 1]
        result["task"] = self.tasks[int(result["task_id"])]["language"]
        return result
