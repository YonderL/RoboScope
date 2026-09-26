"""Continue a Pi-0 LoRA checkpoint on two Pro 5000s in a separate, auditable run."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path


def continuation_config(previous, source, output, checkpoint_hash, micro_batch):
    """Only change placement and batch partitioning; preserve the optimizer trajectory.

    Equal global batches preserve the sampler's per-step sample sets. Different
    microbatch shapes change stochastic noise grouping and floating-point sums,
    so this is a documented continuation, not a bitwise reproduction.
    """
    cfg = copy.deepcopy(previous["config"])
    world, batch = cfg["world_size"], cfg["batch_size"]
    if cfg["policy"] != "pi0_lora" or world != 2:
        raise ValueError("Expected a two-rank Pi-0 LoRA checkpoint")
    if micro_batch <= 0 or batch % (world * micro_batch):
        raise ValueError("micro_batch must divide the unchanged per-rank effective batch")
    if len(previous["rng_by_rank"]) != world or "optimizer" not in previous:
        raise ValueError("Continuation requires optimizer and both rank RNG states")
    cfg.update(
        output_root=str(output),
        gpu_model="pro5000",
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=batch // (world * micro_batch),
        validation_micro_batch_size=cfg.get("validation_micro_batch_size", cfg["micro_batch_size"]),
    )
    cfg["continuation"] = {
        "parent_run": str(source),
        "parent_checkpoint_sha256": checkpoint_hash,
        "parent_step": previous["step"],
        "parent_config": previous["config"],
        "numerics": "New GPU/CUDA and microbatch grouping; not bitwise identical to parent",
    }
    return cfg


def prepare_continuation(source, output, micro_batch):
    import torch

    from roboscope.data.libero import digest
    from roboscope.runtime.training import atomic_checkpoint, manifest_digest
    from roboscope.workflows.experiments import save_contract

    parent = source / "last.pt"
    previous = torch.load(parent, map_location="cpu", weights_only=False)
    manifest = previous["manifest"]
    if previous["manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("Parent checkpoint manifest is inconsistent")
    if json.loads((source / "manifest.json").read_text()) != manifest:
        raise ValueError("Parent run manifest differs from its checkpoint")
    cfg = continuation_config(previous, source, output, digest(parent), micro_batch)
    save_contract(output, cfg, manifest, resume=False)
    # Reuse immutable image arrays. Never rewrite the parent checkpoint or logs.
    if not (source / "image_cache").is_dir():
        raise FileNotFoundError("Missing parent image cache")
    (output / "image_cache").symlink_to(source / "image_cache", target_is_directory=True)
    migrated = {**previous, "config": cfg}
    atomic_checkpoint(output / "last.pt", migrated)
    if (source / "best.pt").is_file():
        best = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        atomic_checkpoint(output / "best.pt", {**best, "config": cfg})
    return cfg


def main():
    from roboscope.data.libero import save_json
    from roboscope.runtime.gpu import gpu_inventory, run_workers
    from roboscope.workflows.experiments import lock_run, save_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Use a separate output outside the parent run")
    if not args.start:
        print(f"Preview: continue {source}/last.pt on two Pro 5000 GPUs into {output}")
        print(f"Microbatch {args.micro_batch}; optimizer, LR schedule and global batch retained")
        return
    # Probe devices before creating any experiment artifacts.
    cards = gpu_inventory("pro5000")
    lock = lock_run(output)
    try:
        if args.resume:
            cfg = json.loads((output / "config.json").read_text())
            if cfg["micro_batch_size"] != args.micro_batch or cfg["continuation"]["parent_run"] != str(
                source
            ):
                raise ValueError("Resume must retain the continuation's parent and microbatch")
            manifest = json.loads((output / "manifest.json").read_text())
            save_contract(output, cfg, manifest, resume=True)
        else:
            cfg = prepare_continuation(source, output, args.micro_batch)
        save_json(output / "gpu_mapping.json", cards)
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(card["uuid"] for card in cards),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS=str(cfg["cpu_threads"]),
            MKL_NUM_THREADS=str(cfg["cpu_threads"]),
            OPENBLAS_NUM_THREADS="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            HF_HUB_OFFLINE="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            # Host transport is conservative and sufficient for small LoRA gradients.
            NCCL_P2P_DISABLE="1",
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
            str(output),
            "--resume",
        ]
        if args.stop_after is not None:
            command += ["--stop-after", str(args.stop_after)]
        save_json(output / "command.json", command)
        print(f"Continuing on GPUs {[c['index'] for c in cards]}; log: {output / 'train.log'}", flush=True)
        run_workers([command], [env], [output / "train.log"])
    finally:
        lock.close()


if __name__ == "__main__":
    main()
