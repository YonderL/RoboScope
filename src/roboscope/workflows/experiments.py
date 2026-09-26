"""Compose datasets, trainers, and evaluation workers without model-specific imports.

Workflows own run directories and provenance. Workers own GPU computation. A dry
run never imports a simulator, touches a GPU, or scans the demonstration dataset.
"""

import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path


def snapshot(run):
    """Freeze the package source and refuse resume after an implementation change."""
    source = Path(__file__).resolve().parents[1]
    archive = run / "source" / "roboscope"
    current = {p.relative_to(source) for p in source.rglob("*.py")}
    if archive.exists():
        saved = {p.relative_to(archive) for p in archive.rglob("*.py")}
        if saved != current:
            raise ValueError("Source file set changed; use a new output directory")
    for relative in sorted(current):
        target = archive / relative
        if target.exists() and target.read_bytes() != (source / relative).read_bytes():
            raise ValueError(f"Source changed: {relative}; use a new output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source / relative, target)


def save_contract(run, cfg, manifest, resume):
    from roboscope.data.libero import save_json

    if (run / "config.json").exists():
        if not resume:
            raise ValueError("Existing run; pass --resume or choose a new output")
        for name, value in [("config.json", cfg), ("manifest.json", manifest)]:
            if json.loads((run / name).read_text()) != value:
                raise ValueError(f"{name} changed; refusing to mix experiments")
    else:
        if any(run.iterdir()):
            # The caller creates only the launcher lock before initialization.
            if set(p.name for p in run.iterdir()) != {".launcher.lock"}:
                raise ValueError("Output is not an empty run directory")
        save_json(run / "config.json", cfg)
        save_json(run / "manifest.json", manifest)
    snapshot(run)
    requirements = run / "requirements.txt"
    if not requirements.exists():
        requirements.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))


def lock_run(run):
    run.mkdir(parents=True, exist_ok=True)
    handle = (run / ".launcher.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def train(cfg, resume=False):
    """ACT schedules independent K/seed runs; DP trains one suite-conditioned model."""
    if cfg["policy"] == "smolvla":
        from roboscope.workflows.smolvla import train_smolvla

        return train_smolvla(cfg, resume)
    if cfg["policy"] == "pi0_lora":
        from roboscope.workflows.pi0 import train_pi0

        return train_pi0(cfg, resume)
    from roboscope.data.cache import prepare_cache
    from roboscope.data.libero import prepare, save_json
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    run = Path(cfg["output_root"])
    lock = lock_run(run)
    try:
        manifest = prepare(cfg)
        if cfg["policy"] == "diffusion":
            from roboscope.data.normalization import add_action_limits

            add_action_limits(manifest)
        save_contract(run, cfg, manifest, resume)
        cards, envs = cards_and_envs()
        save_json(run / "gpu_mapping.json", cards)
        prepare_cache(manifest, run / "image_cache")
        if cfg["policy"] == "diffusion":
            run_workers(
                [[sys.executable, "-m", "roboscope.trainers.diffusion", "--run", str(run)]],
                envs[:1],
                [run / "train.log"],
            )
        elif cfg["policy"] == "act":
            jobs = []
            for seed in cfg["seeds"]:
                for chunk in cfg["chunks"]:
                    folder = run / f"k{chunk:03d}_seed{seed}"
                    folder.mkdir(exist_ok=True)
                    spec = {"config": cfg, "runtime": cfg["runtime"], "chunk": chunk, "seed": seed}
                    path = folder / "run.json"
                    if path.exists() and json.loads(path.read_text()) != spec:
                        raise ValueError(f"Changed ACT spec: {folder}")
                    save_json(path, spec)
                    jobs.append(folder)
            # Cache pretrained weights once, before workers may download concurrently.
            if cfg["pretrained_backbone"]:
                from torchvision.models import ResNet18_Weights

                ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            for start in range(0, len(jobs), 2):
                batch = jobs[start : start + 2]
                run_workers(
                    [
                        [sys.executable, "-m", "roboscope.trainers.act", "train", "--run", str(p)]
                        for p in batch
                    ],
                    envs[: len(batch)],
                    [p / "worker.log" for p in batch],
                )
        else:
            raise ValueError(f"Unsupported trainer: {cfg['policy']}")
    finally:
        lock.close()


def evaluation_config(source, checkpoint, episodes, ddim_steps, ta, rlt_reference=False):
    """Separate evaluation outputs permit comparing checkpoints without overwriting runs."""
    source = Path(source).resolve()
    if (source / "run.json").exists():
        ta = 8 if ta is None else ta
        spec = json.loads((source / "run.json").read_text())
        if spec["chunk"] != 8:
            raise ValueError(
                "Matched ACT-DP evaluator supports ACT K=8; use historical evaluator for other K"
            )
        cfg = dict(spec["config"])
        manifest = json.loads((source.parent / "manifest.json").read_text())
        cfg.update(
            policy="act",
            act_model_config=spec["config"],
            seed=spec["seed"],
            act_checkpoint=str(source / "last.pt"),
            eval_envs=8,
            latency_warmup=5,
            latency_repeats=30,
        )
        variant = "act_k008_chunk"
        if checkpoint != "final" or ta != 8:
            raise ValueError("ACT matched comparison requires --checkpoint final --ta 8")
    else:
        cfg = json.loads((source / "config.json").read_text())
        # Older DP runs kept benchmark paths in a separate environment contract.
        if (source / "env_config.json").exists():
            environment = json.loads((source / "env_config.json").read_text())
            cfg = {**environment, **cfg}
        manifest = json.loads((source / "manifest.json").read_text())
        if cfg.get("policy") == "smolvla_rlt":
            if checkpoint != "final":
                raise ValueError(
                    "RLT does not select checkpoints on evaluation successes; use --checkpoint final"
                )
            ta = cfg["action_horizon"] if ta is None else ta
            if ta != cfg["action_horizon"] or ddim_steps != 10:
                raise ValueError("RLT must use its trained action horizon; DDIM overrides do not apply")
            cfg["evaluation_policy"] = "sft_reference" if rlt_reference else "rlt"
            label = "reference" if rlt_reference else "rlt"
            variant = f"smolvla_{label}_c{ta:03d}"
        elif cfg.get("policy") == "smolvla":
            ta = cfg["action_horizon"] if ta is None else ta
            if ta != cfg["action_horizon"] or ddim_steps != 10:
                raise ValueError(
                    "SmolVLA must use its configured action horizon; --ddim-steps is a DP option"
                )
            variant = f"smolvla_chunk{ta}"
        else:
            cfg["policy"] = "diffusion"
            ta = 8 if ta is None else ta
            if ta not in (1, 4, 8):
                raise ValueError("DP --ta must be 1, 4 or 8")
            variant = f"dp_ddim{ddim_steps:02d}_ta{ta:02d}"
    if rlt_reference and cfg["policy"] != "smolvla_rlt":
        raise ValueError("--rlt-reference requires a SmolVLA RLT run")
    cfg.update(
        eval_episodes=episodes,
        evaluation_checkpoint=checkpoint,
        ddim_steps=ddim_steps,
        action_horizon=ta,
        ddim_ablation=[ddim_steps],
        action_horizon_ablation=[ta],
    )
    for task in manifest["tasks"]:
        task["eval_initial_state_ids"] = list(range(episodes))
    return cfg, manifest, variant


def evaluate(
    source,
    output,
    checkpoint="final",
    episodes=50,
    ddim_steps=10,
    ta=None,
    resume=False,
    rlt_reference=False,
    gpu_model=None,
    task_ids=None,
    libero_root=None,
):
    if gpu_model not in (None, "4090", "5880", "pro5000"):
        raise ValueError(f"Unsupported GPU model: {gpu_model}")
    if task_ids is not None and (
        not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= set(range(10))
    ):
        raise ValueError("Task IDs must be unique integers in 0..9")
    config_path = Path(source) / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()).get("policy") == "pi0_lora":
        if rlt_reference:
            raise ValueError("--rlt-reference requires a SmolVLA RLT run")
        if ddim_steps != 10 or ta not in (None, 8):
            raise ValueError("--ddim-steps and --ta are DP options; Pi-0 uses its saved flow/action horizons")
        from roboscope.workflows.pi0 import evaluate_pi0

        return evaluate_pi0(source, output, checkpoint, episodes, resume, libero_root, gpu_model, task_ids)
    if libero_root is not None:
        raise ValueError("ACT/DP/SmolVLA use benchmark paths from their saved manifest")
    import torch

    from roboscope.data.libero import save_json
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    cfg, manifest, variant = evaluation_config(source, checkpoint, episodes, ddim_steps, ta, rlt_reference)
    run = Path(output).resolve()
    cfg["output_root"] = str(run)
    if task_ids is not None:
        cfg["evaluation_task_ids"] = sorted(task_ids)
    if gpu_model is not None:
        cfg["evaluation_gpu_model"] = gpu_model
    # Fail before recording an invalid initial-state schedule.
    for task in manifest["tasks"]:
        if len(torch.load(task["init"], map_location="cpu", weights_only=False)) < episodes:
            raise ValueError("Insufficient fixed initial states; wrapping is forbidden")
    weight = (
        Path(cfg["act_checkpoint"])
        if variant.startswith("act")
        else Path(source).resolve() / f"{checkpoint}.pt"
    )
    if not weight.is_file():
        raise FileNotFoundError(weight)
    lock = lock_run(run)
    try:
        save_contract(run, cfg, manifest, resume)
        save_json(run / "env_config.json", cfg)
        for name, target in [
            ("image_cache", Path(source).resolve() / "image_cache"),
            (f"{checkpoint}.pt", weight),
        ]:
            if name == "image_cache" and variant.startswith("act"):
                target = Path(source).resolve().parent / name
            if not (run / name).exists():
                (run / name).symlink_to(target)
        cards, envs = cards_and_envs(gpu_model or "4090")
        save_json(run / "gpu_mapping.json", cards)
        commands = [
            [
                sys.executable,
                "-m",
                "roboscope.evaluation.worker",
                "--run",
                str(run),
                "--variant",
                variant,
                "--shard",
                str(i),
            ]
            for i in range(2)
        ]
        logs = [run / f"eval_shard{i}.log" for i in range(2)]
        if cfg.get("policy") == "smolvla" and gpu_model is None:
            # One 4090: finish even-task shard, then odd-task shard, on the last card.
            for command, log in zip(commands, logs, strict=True):
                run_workers([command], [envs[-1]], [log])
        else:
            run_workers(commands, envs, logs)
        from roboscope.reporting.records import read_evaluation

        rows, identity = read_evaluation(run / "eval" / variant, episodes, task_ids)
        save_json(
            run / "summary.json",
            {**identity, "episodes": len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows)},
        )
    finally:
        lock.close()
