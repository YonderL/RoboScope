"""Prepare a separate, immutable RLT run from an existing SFT checkpoint."""

import json
import sys
from pathlib import Path

from roboscope.workflows.experiments import lock_run, save_contract


def posttrain(source, output, cfg, checkpoint="final", resume=False, stage="all"):
    import torch

    from roboscope.data.cache import prepare_cache
    from roboscope.data.libero import digest, save_json
    from roboscope.runtime.gpu import cards_and_envs, run_workers
    from roboscope.trainers.pi0 import manifest_digest

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("RLT must use a separate output directory from SFT")
    source_cfg = json.loads((source / "config.json").read_text())
    if source_cfg.get("policy") != "smolvla":
        raise ValueError("--source must be a SmolVLA SFT run")
    weight = source / f"{checkpoint}.pt"
    manifest = json.loads((source / "manifest.json").read_text())
    saved = torch.load(weight, map_location="cpu", weights_only=False)
    if saved.get("format") != "roboscope.smolvla.v1" or saved.get("manifest_sha256") != manifest_digest(
        manifest
    ):
        raise ValueError("SFT checkpoint and manifest do not match")
    del saved
    for task in manifest["tasks"]:
        if digest(task["bddl"]) != task["bddl_sha256"] or digest(task["init"]) != task["init_sha256"]:
            raise ValueError("LIBERO assets changed since SFT")
    cfg = {
        **cfg,
        "sft_checkpoint": str(weight),
        "sft_sha256": digest(weight),
        "sft_source": str(source),
        "data_root": source_cfg["data_root"],
        "libero_root": source_cfg["libero_root"],
        "output_root": str(output),
    }
    lock = lock_run(output)
    try:
        save_contract(output, cfg, manifest, resume)
        cache = output / "image_cache"
        if not cache.exists():
            if (source / "image_cache").is_dir():
                cache.symlink_to(source / "image_cache")
            else:
                prepare_cache(manifest, cache)
        cards, envs = cards_and_envs()
        save_json(output / "gpu_mapping.json", cards)
        command = [
            sys.executable,
            "-m",
            "roboscope.trainers.smolvla_rlt",
            "--run",
            str(output),
            "--stage",
            stage,
        ]
        if resume:
            command.append("--resume")
        envs[0]["TOKENIZERS_PARALLELISM"] = "false"
        run_workers([command], envs[:1], [output / f"train_{stage}.log"])
    finally:
        lock.close()
