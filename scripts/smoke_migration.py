"""Opt-in, bounded migration regression against a local archived experiment.

Two RTX 4090 workers: one checks ACT, one checks DP. Each compares predictions
against archived model code on identical observations, then runs at most two
three-step simulator episodes. No optimizer updates or formal scores are saved.
"""

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--worker", choices=["act", "dp"])
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    archive = a.archive.resolve()
    if not a.worker:
        from roboscope.runtime.gpu import cards_and_envs, run_workers

        _, envs = cards_and_envs()
        with tempfile.TemporaryDirectory(prefix="roboscope-smoke-") as folder:
            folder = Path(folder)
            commands = [
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--archive",
                    str(archive),
                    "--worker",
                    kind,
                    "--report",
                    str(folder / f"{kind}.json"),
                ]
                for kind in ["act", "dp"]
            ]
            run_workers(commands, envs, [folder / f"{kind}.log" for kind in ["act", "dp"]])
            result = {kind: json.loads((folder / f"{kind}.json").read_text()) for kind in ["act", "dp"]}
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    import torch

    from roboscope.data.sequences import SequenceDataset
    from roboscope.evaluation.worker import rollout
    from roboscope.runtime.common import require_4090, seed_all

    require_4090()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    seed_all(0)
    cfg = json.loads((archive / "config.json").read_text())
    manifest = json.loads((archive / "manifest.json").read_text())
    env_cfg = json.loads((archive / "env_config.json").read_text())
    ds = SequenceDataset(manifest, "val", archive / "image_cache")
    batch = torch.utils.data.default_collate([ds[0], ds[len(ds) // 2]])
    batch = {k: v.cuda() for k, v in batch.items() if k not in ("action", "action_is_pad")}
    # Archived imports stay inside this diagnostic process; production imports do not depend on them.
    sys.path.insert(0, str(archive / "source"))
    if a.worker == "act":
        from roboscope.policies.act import TaskACT

        old = module(archive / "source/act_reference/model.py", "reference_act").TaskACT
        old_model = old(env_cfg, manifest, 8, initialize_backbone=False)
        new_model = TaskACT(env_cfg, manifest, 8, initialize_backbone=False)
        path = archive / "act_final.pt"
        for k in ("state", "agentview_rgb", "eye_in_hand_rgb"):
            batch[k] = batch[k][:, -1]
    else:
        from roboscope.policies.diffusion import TaskDiffusionPolicy

        old = module(archive / "source/policy.py", "reference_diffusion").TaskDiffusionPolicy
        old_model = old(cfg, manifest)
        new_model = TaskDiffusionPolicy(cfg, manifest)
        path = archive / "final.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old_model.load_state_dict(ckpt["model"])
    new_model.load_state_dict(ckpt["model"])
    del ckpt
    old_model.cuda().eval()
    new_model.cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if a.worker == "dp":
            noise = torch.randn(2, 16, 7, device="cuda")
            before = old_model.predict(batch, 10, noise.clone())
            after = new_model.predict(batch, 10, noise.clone())
        else:
            before = old_model.predict(batch)
            after = new_model.predict(batch)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    max_diff = (before - after).abs().max().item()
    del old_model
    variant = {"model": a.worker, "ta": 8, "ddim_steps": 10}
    small = {**cfg, "eval_envs": 2, "rollout_horizon": 3, "video_episodes_per_task": 1}
    with tempfile.TemporaryDirectory(prefix="roboscope-rollout-") as folder:
        folder = Path(folder)
        rows = []
        rollout(
            new_model,
            small,
            {**env_cfg, "rollout_horizon": 3},
            [(manifest["tasks"][0], i) for i in [0, 1]],
            variant,
            folder,
            rows.append,
        )
        assert len(rows) == 2 and all(r["steps"] <= 3 for r in rows)
        assert len(list(folder.glob("*.npz"))) == 2 and len(list(folder.glob("*.mp4"))) == 1
    a.report.write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "gpu": torch.cuda.get_device_name(0),
                "checkpoint_step": 30000 if a.worker == "dp" else 56064,
                "max_prediction_difference": max_diff,
                "smoke_episodes": 2,
                "max_steps_per_episode": 3,
                "video_and_trace_written": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
