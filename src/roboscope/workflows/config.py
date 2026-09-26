"""Reject silent recipe changes that the current implementation cannot execute."""


def validate_recipe(cfg):
    if cfg.get("policy") == "smolvla":
        validate_smolvla(cfg)
        return
    if cfg.get("policy") == "pi0_lora":
        validate_pi0(cfg)
        return
    if cfg.get("policy") not in ("act", "diffusion"):
        raise ValueError("Only ACT, Diffusion Policy, SmolVLA and Pi-0 LoRA trainers are implemented")
    for key in ("batch_size", "cpu_threads", "eval_episodes", "rollout_horizon"):
        if cfg.get(key, 0) < 1:
            raise ValueError(f"{key} must be positive")
    if cfg.get("workers", -1) < 0 or cfg.get("settle_steps", -1) < 0:
        raise ValueError("workers and settle_steps must be nonnegative")
    if not 0 < cfg.get("validation_fraction", 0) < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if cfg["policy"] == "diffusion":
        expected = {
            "obs_horizon": 2,
            "prediction_horizon": 16,
            "diffusion_steps": 100,
            "ddim_eta": 0,
            "clip_sample": True,
            "action_normalization": "minmax",
            "backbone_norm": "groupnorm",
            "pretrained_backbone": False,
            "share_rgb_model": False,
            "crop_shape": [116, 116],
            "task_setting": "suite_task_id",
            "prediction_type": "epsilon",
        }
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError(f"This DP recipe requires {key}={value!r}")
        if cfg["train_steps"] < 1 or cfg["validate_every"] < 1 or cfg["warmup_steps"] < 1:
            raise ValueError("Training steps and schedule intervals must be positive")
        if not 1 <= cfg["action_horizon"] <= 15:
            raise ValueError("Invalid action horizon")
    else:
        for key in ("epochs", "eval_every"):
            if cfg.get(key, 0) < 1:
                raise ValueError(f"{key} must be positive")
        if (
            not cfg["chunks"]
            or any(k < 1 for k in cfg["chunks"])
            or len(set(cfg["chunks"])) != len(cfg["chunks"])
        ):
            raise ValueError("chunks must be distinct positive integers")
        if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
            raise ValueError("seeds must be distinct")
        if cfg["modes"] != ["chunk", "replan", "ensemble"]:
            raise ValueError("ACT recipe requires all three execution modes")


def validate_smolvla(cfg):
    for key in (
        "batch_size",
        "micro_batch_size",
        "train_steps",
        "cpu_threads",
        "validate_every",
        "validation_samples",
        "save_every",
        "log_every",
        "eval_episodes",
        "eval_envs",
        "rollout_horizon",
        "latency_repeats",
    ):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("workers", "settle_steps", "latency_warmup", "video_episodes_per_task"):
        if type(cfg.get(key)) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg["micro_batch_size"] > cfg["batch_size"]:
        raise ValueError("micro_batch_size must not exceed batch_size")
    if not 0 < cfg.get("validation_fraction", 0) < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for key, value in {"chunk_size": 50, "task_setting": "natural_language"}.items():
        if cfg.get(key) != value:
            raise ValueError(f"SmolVLA requires {key}={value!r}")
    if not 1 <= cfg.get("action_horizon", 0) <= cfg["chunk_size"]:
        raise ValueError("action_horizon must be within the 50-step SmolVLA chunk")
    for key in ("pretrained_path", "pretrained_revision", "vlm_path", "vlm_revision"):
        if not cfg.get(key):
            raise ValueError(f"{key} is required")


def validate_rlt(cfg):
    if cfg.get("policy") != "smolvla_rlt":
        raise ValueError("posttrain requires policy=smolvla_rlt")
    if cfg.get("token_features") != "vlm_image_tokens":
        raise ValueError(
            "RLT requires token_features=vlm_image_tokens; old full-prefix tokens need retraining"
        )
    integers = (
        "action_horizon",
        "token_dim",
        "token_layers",
        "token_heads",
        "token_steps",
        "token_batch_size",
        "token_save_every",
        "feature_batch_size",
        "policy_delay",
        "warmup_steps",
        "online_steps",
        "updates_per_transition",
        "batch_size",
        "replay_capacity",
        "replay_stride",
        "rollout_horizon",
        "cpu_threads",
        "log_every",
        "eval_episodes",
        "eval_envs",
        "latency_repeats",
    )
    for key in integers:
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "workers",
        "settle_steps",
        "initial_updates",
        "latency_warmup",
        "video_episodes_per_task",
        "seed",
        "eval_seed",
        "rollout_seed",
    ):
        if type(cfg.get(key)) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not 1 <= cfg["action_horizon"] <= 50:
        raise ValueError("RL action_horizon must be within SmolVLA's 50-step proposal")
    if cfg["replay_stride"] > cfg["action_horizon"]:
        raise ValueError("replay_stride must not exceed action_horizon")
    if cfg["token_dim"] % cfg["token_heads"] or cfg["token_dim"] % 2:
        raise ValueError("token_dim must be even and divisible by token_heads")
    if cfg["warmup_steps"] >= cfg["online_steps"] or cfg["replay_capacity"] < cfg["batch_size"]:
        raise ValueError("Need an online budget beyond warmup and replay capacity >= batch_size")
    if not 0 < cfg.get("gamma", 0) <= 1 or not 0 < cfg.get("target_tau", 0) <= 1:
        raise ValueError("gamma and target_tau must be in (0,1]")
    if not 0 <= cfg.get("reference_dropout", -1) <= 1:
        raise ValueError("reference_dropout must be in [0,1]")
    for key in ("token_lr", "actor_lr", "critic_lr", "grad_clip", "actor_std"):
        if cfg.get(key, 0) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.get("bc_weight", -1) < 0:
        raise ValueError("bc_weight must be nonnegative")
    if not cfg.get("hidden_dims") or any(type(v) is not int or v < 1 for v in cfg["hidden_dims"]):
        raise ValueError("hidden_dims must contain positive integers")


def validate_pi0(cfg):
    for key in (
        "batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "train_steps",
        "cpu_threads",
        "validate_every",
        "validation_samples",
        "save_every",
        "log_every",
        "eval_episodes",
        "rollout_horizon",
        "lora_rank",
        "num_inference_steps",
        "tokenizer_max_length",
    ):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    expected = {
        "world_size": 2,
        "chunk_size": 50,
        "max_action_dim": 32,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "task_setting": "natural_language",
        "mask_padding_loss": False,
        "lora_dropout": 0.0,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Pi-0 LoRA requires {key}={value!r}")
    if cfg["batch_size"] != 2 * cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"]:
        raise ValueError("batch_size must equal 2 * micro_batch_size * gradient_accumulation_steps")
    if not 1 <= cfg.get("action_horizon", 0) <= cfg["chunk_size"]:
        raise ValueError("action_horizon must be within chunk_size")
    if not 0 < cfg.get("validation_fraction", 0) < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if cfg.get("workers", -1) < 0 or cfg.get("settle_steps", -1) < 0:
        raise ValueError("workers and settle_steps must be nonnegative")
    if not 0 <= cfg.get("warmup_steps", -1) < cfg["train_steps"]:
        raise ValueError("warmup_steps must be nonnegative and less than train_steps")
    if not 0 < cfg.get("min_lr_ratio", 0) <= 1:
        raise ValueError("min_lr_ratio must be in (0,1]")
    for key in ("lr", "grad_clip", "lora_alpha"):
        if cfg.get(key, 0) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.get("weight_decay", -1) < 0:
        raise ValueError("weight_decay must be nonnegative")
    if not cfg.get("pretrained_path") or not cfg.get("tokenizer_path"):
        raise ValueError("pretrained_path and tokenizer_path are required")
