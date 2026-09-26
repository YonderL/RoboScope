"""A separate paper-based Spatial experiment; old ACT/DP/SFT runs keep their adapters."""

import argparse
import json
import sys
from pathlib import Path

from roboscope.data.smolvla_official import DATASET_ID, DATASET_REVISION


def build_train_config(recipe, dataset, vlm, output, smoke=False):
    from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if recipe["precision"] != "bf16":
        raise ValueError("This experiment uses BF16 autocast")
    # No policy.pretrained_path: only the VLM is pretrained; expert and robot
    # projections initialize randomly. Loading smolvla_base would change this experiment.
    policy = SmolVLAConfig(
        device="cpu",
        push_to_hub=False,
        use_amp=False,
        vlm_model_name=str(vlm),
        load_vlm_weights=True,
        chunk_size=recipe["chunk_size"],
        n_action_steps=recipe["n_action_steps"],
        num_steps=recipe["num_steps"],
        num_vlm_layers=recipe["num_vlm_layers"],
        expert_width_multiplier=recipe["expert_width_multiplier"],
        train_expert_only=True,
        freeze_vision_encoder=True,
        train_state_proj=True,
        prefix_length=0,
        pad_language_to="max_length",
        tokenizer_max_length=48,
        scheduler_warmup_steps=recipe["scheduler_warmup_steps"],
        scheduler_decay_steps=recipe["scheduler_decay_steps"],
        compile_model=recipe["compile_model"],
        compile_mode=recipe["compile_mode"],
    )
    # Native config construction silently falls back to CPU when the launcher
    # cannot see CUDA. Serialize the worker's intended settings explicitly;
    # the UUID-bound worker checks require_4090() before parsing this config.
    policy.device = "cuda"
    policy.use_amp = True
    env = LiberoEnv(
        task="libero_spatial",
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        max_parallel_tasks=1,
        task_ids=[0] if smoke else None,
        episode_length=10 if smoke else None,
    )
    # One task at a time, but all 10 episodes of that task run together.
    # Sync n_envs=1 left the 4090 near 4GB and about 6 sim steps/s.
    eval_envs = 1 if smoke else recipe["eval_episodes"]
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"{DATASET_ID}_spatial", root=str(dataset), eval_split=0.0),
        policy=policy,
        env=env,
        output_dir=output,
        seed=recipe["seed"],
        job_name="smolvla_paper_spatial_smoke" if smoke else "smolvla_paper_spatial",
        steps=2 if smoke else recipe["steps"],
        batch_size=recipe["batch_size"],
        num_workers=recipe["num_workers"],
        prefetch_factor=2,
        save_freq=2 if smoke else recipe["save_freq"],
        log_freq=1 if smoke else recipe["log_freq"],
        env_eval_freq=2 if smoke else recipe["env_eval_freq"],
        eval=EvalConfig(
            n_episodes=1 if smoke else recipe["eval_episodes"],
            batch_size=eval_envs,
            use_async_envs=eval_envs > 1,
        ),
        wandb=WandBConfig(enable=False),
    )


def build_eval_config(train_cfg, recipe, run, smoke=False):
    native = train_cfg.to_dict()
    # Paper section 4.1 uses 10 trials/task. Keep the final sample count explicit
    # so a larger follow-up evaluation need not alter periodic training checks.
    native["eval"]["n_episodes"] = 1 if smoke else recipe["final_eval_episodes"]
    return {
        "env": native["env"],
        "eval": native["eval"],
        "seed": recipe["seed"],
        "output_dir": str(run / "evaluation_final"),
        "job_name": "smolvla_paper_spatial_final",
    }


def prepare_run(recipe, run, dataset, cache, libero_root, smoke):
    from huggingface_hub import snapshot_download

    from roboscope.data.libero import save_json
    from roboscope.workflows.experiments import save_contract

    if recipe["dataset_repo"] != DATASET_ID or recipe["dataset_revision"] != DATASET_REVISION:
        raise ValueError("The recipe and prepared dataset must use the pinned official release")
    provenance = json.loads((dataset / "spatial_provenance.json").read_text())
    if provenance["source_revision"] != recipe["dataset_revision"]:
        raise ValueError("Dataset revision mismatch")
    vlm = Path(
        snapshot_download(
            recipe["vlm_repo"],
            revision=recipe["vlm_revision"],
            cache_dir=cache / "hub",
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja", "*.safetensors"],
            max_workers=4,
        )
    )
    effective = {
        **recipe,
        "smoke": smoke,
        "dataset_root": str(dataset),
        "vlm_path": str(vlm),
        "libero_root": str(libero_root),
    }
    save_contract(run, effective, provenance, resume=(run / "config.json").exists())
    cfg = build_train_config(recipe, dataset, vlm, run / "training", smoke)
    config_dir = run / "configs"
    config_dir.mkdir(exist_ok=True)
    cfg.save_pretrained(config_dir)
    # Use local benchmark assets. Sync evaluation keeps the asset override in
    # the worker process; the native processor supplies 8D state and image rotation.
    import yaml

    env_dir = run / "libero_config"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(libero_root),
                "bddl_files": str(libero_root / "bddl_files"),
                "init_states": str(libero_root / "init_files"),
                "assets": str(libero_root / "assets"),
                "datasets": str(dataset),
            }
        )
    )
    save_json(config_dir / "eval_config.json", build_eval_config(cfg, recipe, run, smoke))
    return config_dir


def worker_command(run, configs, libero_root, stage, resume=False):
    """Resume uses the checkpoint's optimizer/RNG config, not a new train config."""
    checkpoint = run / "training/checkpoints/last/pretrained_model"
    config = configs / "train_config.json"
    command = [sys.executable, "-u", "-m", "roboscope.trainers.smolvla_official"]
    if stage == "evaluate":
        if resume:
            raise ValueError("--resume is for training, not standalone evaluation")
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError("No native LeRobot checkpoint available to evaluate")
        config = configs / "eval_config.json"
        command += ["--evaluate", "--checkpoint", str(checkpoint)]
    elif resume:
        config = checkpoint / "train_config.json"
        if not config.is_file():
            raise FileNotFoundError("No native LeRobot checkpoint available to resume")
        command.append("--resume")
    elif (run / "training").exists():
        raise FileExistsError("Training output exists; use --resume or a new output directory")
    return command + ["--config", str(config), "--libero-root", str(libero_root)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "smoke", "train", "evaluate"], default="train")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    print(json.dumps({"stage": args.stage, "recipe": recipe, "output": str(args.output)}, indent=2))
    if not args.start:
        print("Preview only. --start uses the last RTX 4090, with no other GPU workers.")
        return
    from roboscope.data.libero import save_json
    from roboscope.runtime.gpu import cards_and_envs, run_workers
    from roboscope.workflows.experiments import lock_run

    run, dataset, cache, libero_root = (
        p.resolve() for p in (args.output, args.dataset, args.cache, args.libero_root)
    )
    lock = lock_run(run)
    try:
        smoke = args.stage == "smoke"
        if args.stage == "evaluate" and (run / "config.json").exists():
            smoke = json.loads((run / "config.json").read_text())["smoke"]
        configs = prepare_run(recipe, run, dataset, cache, libero_root, smoke)
        if args.stage == "prepare":
            return
        from roboscope.data.smolvla_image_cache import (
            build_image_cache,
            build_numeric_cache,
            cache_dir_for,
        )

        image_cache = build_image_cache(dataset, cache_dir_for(dataset))
        build_numeric_cache(dataset, image_cache)
        command = worker_command(run, configs, libero_root, args.stage, args.resume)
        cards, envs = cards_and_envs()
        save_json(run / "gpu_mapping.json", cards[-1:])
        env = envs[-1]
        env.update(
            TOKENIZERS_PARALLELISM="false",
            LIBERO_CONFIG_PATH=str(run / "libero_config"),
            HF_HOME=str(cache),
            HF_DATASETS_CACHE=str(cache / "datasets"),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
            HF_HUB_OFFLINE="1",
            ROBOSCOPE_IMAGE_CACHE=str(image_cache),
        )
        save_json(run / f"command_{args.stage}.json", command)
        run_workers([command], [env], [run / f"{args.stage}.log"])
    finally:
        lock.close()


if __name__ == "__main__":
    main()
