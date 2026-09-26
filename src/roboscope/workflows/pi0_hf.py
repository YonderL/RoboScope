"""Train Pi-0 LoRA on the extracted HuggingFaceVLA Spatial set, separate from the raw run."""

import argparse
import json
import os
import sys
from pathlib import Path

from roboscope.workflows.experiments import lock_run, save_contract

LANGUAGES = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]


def train(cfg, resume=False):
    from roboscope.data.libero import save_json
    from roboscope.data.pi0_lerobot import prepare_lerobot_manifest
    from roboscope.policies.pi0 import resolve_assets
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    run = Path(cfg["output_root"])
    if (
        Path(cfg["dataset_root"]).resolve() == run.resolve()
        or run.resolve() in Path(cfg["dataset_root"]).resolve().parents
    ):
        raise ValueError("Keep the training run outside the dataset directory")
    lock = lock_run(run)
    try:
        cards, _ = cards_and_envs(cfg["gpu_model"])
        asset_keys = ("pretrained_path", "pretrained_revision", "tokenizer_path", "tokenizer_revision")
        requested = {key: cfg[key] for key in asset_keys}
        if (run / "config.json").exists():
            if not resume:
                raise ValueError("Existing Pi-0 run; pass --resume or choose another output")
            saved = json.loads((run / "config.json").read_text())
            if saved["asset_requests"] != requested:
                raise ValueError("Pretrained asset request changed; use a new output")
            cfg = dict(cfg)
            cfg.update({key: saved[key] for key in asset_keys})
            manifest = json.loads((run / "manifest.json").read_text())
        else:
            cfg = dict(cfg)
            cfg.update(resolve_assets(cfg))
            cache = run.parent / f"{run.name}_cache"
            print(f"Caching {cfg['dataset_root']} into {cache}", flush=True)
            manifest = prepare_lerobot_manifest(
                cfg["dataset_root"],
                cache,
                LANGUAGES,
                cfg["validation_fraction"],
                cfg["split_seed"],
            )
        cfg["asset_requests"] = requested
        save_contract(run, cfg, manifest, resume)
        save_json(run / "gpu_mapping.json", cards)
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=",".join(card["uuid"] for card in cards),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            OMP_NUM_THREADS=str(cfg["cpu_threads"]),
            MKL_NUM_THREADS=str(cfg["cpu_threads"]),
            OPENBLAS_NUM_THREADS="1",
            NCCL_P2P_DISABLE="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "-m",
            "roboscope.trainers.pi0",
            "--run",
            str(run),
        ]
        if resume:
            command.append("--resume")
        print(f"Pi-0 HF Spatial on GPUs {[c['index'] for c in cards]}; log: {run / 'train.log'}", flush=True)
        run_workers([command], [environment], [run / "train.log"])
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not args.start:
        print(f"Preview: {cfg['output_root']} from {cfg['dataset_root']} on {cfg['gpu_model']}")
        return
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
