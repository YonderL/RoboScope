"""Reject silent recipe changes that the current implementation cannot execute."""


def validate_recipe(cfg):
    if cfg.get("policy") not in ("act", "diffusion"):
        raise ValueError("Only ACT and Diffusion Policy trainers are implemented")
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
