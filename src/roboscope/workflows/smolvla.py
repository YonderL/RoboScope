"""SmolVLA preparation and launch; evaluation uses the shared ACT/DP worker."""

import json
import sys
from pathlib import Path

from roboscope.workflows.experiments import lock_run, save_contract


def train_smolvla(cfg, resume=False):
    from roboscope.data.cache import prepare_cache
    from roboscope.data.libero import save_json
    from roboscope.data.smolvla import prepare_smolvla
    from roboscope.policies.smolvla import ASSET_KEYS, resolve_assets
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    run = Path(cfg["output_root"])
    lock = lock_run(run)
    try:
        cfg = dict(cfg)
        cards, envs = cards_and_envs()
        manifest = prepare_smolvla(cfg)
        requested = {key: cfg[key] for key in ASSET_KEYS}
        if (run / "config.json").exists():
            if not resume:
                raise ValueError("Existing SmolVLA run; pass --resume or choose a new output")
            saved = json.loads((run / "config.json").read_text())
            if saved["asset_requests"] != requested:
                raise ValueError("Pretrained assets changed; use a new output")
            cfg.update({key: saved[key] for key in ASSET_KEYS})
        else:
            cfg.update(resolve_assets(cfg))
        cfg["asset_requests"] = requested
        save_contract(run, cfg, manifest, resume)
        save_json(run / "gpu_mapping.json", cards)
        prepare_cache(manifest, run / "image_cache")
        command = [sys.executable, "-m", "roboscope.trainers.smolvla", "--run", str(run)]
        if resume:
            command.append("--resume")
        envs[0]["TOKENIZERS_PARALLELISM"] = "false"
        run_workers([command], envs[:1], [run / "train.log"])
    finally:
        lock.close()
