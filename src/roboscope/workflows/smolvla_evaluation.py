"""Standalone native SmolVLA evaluation with an explicit task/GPU/physics contract."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Native SmolVLA training run")
    parser.add_argument("--checkpoint", default="100000")
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--gpu-model", choices=["4090", "5880", "pro5000"], default="pro5000")
    parser.add_argument("--mujoco-version", default="3.3.2")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if (
        not args.task_ids
        or len(set(args.task_ids)) != len(args.task_ids)
        or not set(args.task_ids) <= set(range(10))
        or args.episodes < 1
    ):
        parser.error("Expected distinct task IDs in 0..9 and positive episodes")
    print(json.dumps(vars(args), default=str, indent=2))
    if not args.start:
        print("Preview only; no model, data or GPU loaded.")
        return

    from importlib.metadata import version

    import yaml

    from roboscope.data.libero import digest, save_json
    from roboscope.reporting.native import native_episodes
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    if version("mujoco") != args.mujoco_version:
        raise ValueError("Installed MuJoCo does not match the requested evaluation protocol")
    source, output, root = (p.resolve() for p in (args.source, args.output, args.libero_root))
    checkpoint = source / "training/checkpoints" / args.checkpoint / "pretrained_model"
    policy = json.loads((checkpoint / "config.json").read_text())
    if not 1 <= args.action_steps <= policy["chunk_size"]:
        raise ValueError("Execution horizon exceeds the predicted chunk")
    config = json.loads((source / "configs/eval_config.json").read_text())
    config["env"]["task_ids"] = sorted(args.task_ids)
    config["eval"].update(n_episodes=args.episodes, batch_size=min(10, args.episodes), use_async_envs=True)
    config["output_dir"] = str(output / "evaluation")
    cards, envs = cards_and_envs(args.gpu_model)
    # Create the local asset configuration before importing LIBERO (which otherwise prompts).
    output.mkdir(parents=True, exist_ok=False)
    asset_config = output / "libero_config"
    asset_config.mkdir()
    (asset_config / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(root),
                "bddl_files": str(root / "bddl_files"),
                "init_states": str(root / "init_files"),
                "assets": str(root / "assets"),
                "datasets": str(source),
            }
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(asset_config)
    # Resolve benchmark ordering from the native evaluator, rather than using
    # the HDF5 manifest's lexicographic IDs (on-ramekin is native T5, HDF5 T7).
    from libero.libero.benchmark import get_benchmark_dict

    suite = get_benchmark_dict()["libero_spatial"]()
    tasks = [suite.get_task(i) for i in args.task_ids]
    identity = {
        "checkpoint_step": args.checkpoint,
        "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
        "checkpoint_files": {p.name: digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
        "task_ids": sorted(args.task_ids),
        "task_names": [task.name for task in tasks],
        "episodes_per_task": args.episodes,
        "action_steps": args.action_steps,
        "environment": {
            name: version(name) for name in ("torch", "mujoco", "robosuite", "hf-libero", "lerobot")
        },
        "gpu": cards[0],
        "evaluation_config": config,
    }
    save_json(output / "identity.json", identity)
    staged = output / "checkpoint"
    staged.mkdir()
    for path in checkpoint.iterdir():
        if path.is_file() and path.name != "config.json":
            (staged / path.name).symlink_to(path)
    policy["n_action_steps"] = args.action_steps
    policy["compile_model"] = False
    save_json(staged / "config.json", policy)
    save_json(output / "eval_config.json", config)
    environment = envs[0]
    environment.update(
        ROBOSCOPE_GPU_MODEL=args.gpu_model,
        LIBERO_CONFIG_PATH=str(asset_config),
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "roboscope.trainers.smolvla_official",
        "--evaluate",
        "--checkpoint",
        str(staged),
        "--config",
        str(output / "eval_config.json"),
        "--libero-root",
        str(root),
    ]
    save_json(output / "command.json", command)
    run_workers([command], [environment], [output / "evaluation.log"])
    report = json.loads((output / "evaluation/eval_info.json").read_text())
    rows = native_episodes([report], args.task_ids, args.episodes)
    if digest(checkpoint / "model.safetensors") != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    save_json(
        output / "summary.json",
        {
            "successes": sum(r["success"] for r in rows),
            "episodes": len(rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "checkpoint_sha256": identity["checkpoint_sha256"],
        },
    )


if __name__ == "__main__":
    main()
