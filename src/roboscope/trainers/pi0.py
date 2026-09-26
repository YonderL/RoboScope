"""Pi-0 LoRA DDP worker, with optimizer-step sampling and adapter-only exports.

Launch with torchrun through the Pi-0 workflow. The global batch is
micro_batch_size * gradient_accumulation_steps * world_size. Validation is a
fixed-sample flow-matching loss; it must never be labelled a task success rate.
"""

import argparse
import csv
import io
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from roboscope.runtime.training import (
    atomic_checkpoint as atomic_checkpoint,
)
from roboscope.runtime.training import (
    atomic_write as atomic_write,
)
from roboscope.runtime.training import (
    capture_rng as capture_rng,
)
from roboscope.runtime.training import (
    cpu_tree as cpu_tree,
)
from roboscope.runtime.training import (
    fixed_rng as fixed_rng,
)
from roboscope.runtime.training import (
    manifest_digest as manifest_digest,
)
from roboscope.runtime.training import (
    move_batch as move_batch,
)
from roboscope.runtime.training import (
    restore_rng as restore_rng,
)
from roboscope.runtime.training import (
    save_json as save_json,
)
from roboscope.runtime.training import (
    seed_all as seed_all,
)


class StepMicroBatchSampler:
    """Partition an absolute, shuffled sample stream into rank-local microbatches.

    All frames are used, including the end of each permutation. Prefetching
    cannot move the resume cursor: only the completed optimizer step is saved.
    Dataset access must be deterministic (Pi0Dataset performs no augmentation).
    """

    def __init__(
        self, size, micro_batch_size, accumulation_steps, steps, seed, rank=0, world_size=1, start_step=0
    ):
        if min(size, micro_batch_size, accumulation_steps, world_size) <= 0:
            raise ValueError("Dataset and batch dimensions must be positive")
        if not 0 <= rank < world_size or not 0 <= start_step <= steps:
            raise ValueError("Invalid rank or resume step")
        self.size, self.micro, self.accumulation = size, micro_batch_size, accumulation_steps
        self.steps, self.seed, self.rank, self.world, self.start = steps, seed, rank, world_size, start_step

    def __len__(self):
        return (self.steps - self.start) * self.accumulation

    def __iter__(self):
        local_batch = self.micro * self.accumulation
        global_batch = local_batch * self.world
        cached_epoch, order = None, None
        for step in range(self.start, self.steps):
            cursor = step * global_batch + self.rank * local_batch
            indices = []
            while len(indices) < local_batch:
                epoch, offset = divmod(cursor, self.size)
                if epoch != cached_epoch:
                    order = torch.randperm(
                        self.size, generator=torch.Generator().manual_seed(self.seed + epoch)
                    ).tolist()
                    cached_epoch = epoch
                count = min(local_batch - len(indices), self.size - offset)
                indices.extend(order[offset : offset + count])
                cursor += count
            for offset in range(0, local_batch, self.micro):
                yield indices[offset : offset + self.micro]


def accumulate_gradients(model, batches, accumulation_steps, device):
    """Mean sample loss: divide once by accumulation; DDP averages ranks itself."""
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for micro in range(accumulation_steps):
        batch = move_batch(next(batches), device)
        sync = (
            model.no_sync() if hasattr(model, "no_sync") and micro + 1 < accumulation_steps else nullcontext()
        )
        # no_sync must surround both the forward pass and backward pass.
        with sync:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(batch)
            if loss.ndim != 0:
                raise ValueError("Pi0Policy.forward must return a scalar batch-mean loss")
            (loss / accumulation_steps).backward()
        loss_sum += loss.detach().double() / accumulation_steps
    return loss_sum


def learning_rate(step, cfg):
    """Learning rate for the next zero-based optimizer step; exact on resume."""
    warmup = cfg["warmup_steps"]
    if step < warmup:
        fraction = (step + 1) / max(1, warmup)
    else:
        progress = min(1.0, (step - warmup) / max(1, cfg["train_steps"] - warmup - 1))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        fraction = cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * cosine
    return cfg["lr"] * fraction


def validation(model, loader, seed, device, rank=0):
    """Sample-weighted global mean with fixed noise and isolated training RNG."""
    was_training = model.training
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    try:
        with fixed_rng(seed + 98765 + rank, device), torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(batch)
                count = len(batch["state"])
                total[0] += loss.double() * count
                total[1] += count
        if dist.is_initialized():
            dist.all_reduce(total)
        if total[1].item() == 0:
            raise ValueError("The validation split is empty")
        score = (total[0] / total[1]).item()
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation loss")
        return score
    finally:
        model.train(was_training)


def write_history(run, history):
    """Rewrite committed metrics on resume, removing any uncheckpointed tail."""
    save_json(run / "train_history.json", history)
    jsonl = "".join(json.dumps(row, allow_nan=False) + "\n" for row in history).encode()
    atomic_write(run / "train_metrics.jsonl", lambda handle: handle.write(jsonl))
    output = io.StringIO(newline="")
    if history:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    csv_bytes = output.getvalue().encode()
    atomic_write(run / "train_metrics.csv", lambda handle: handle.write(csv_bytes))


def append_metrics(run, row, history):
    history.append(row)
    with (run / "train_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    with (run / "train_metrics.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if len(history) == 1:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, allow_nan=False), flush=True)


def export_checkpoint(model, cfg, manifest, step, score, validation_step):
    return {
        "format": "roboscope.pi0_lora.v1",
        "adapter": cpu_tree(model.trainable_state_dict()),
        "step": step,
        "config": cfg,
        "manifest": manifest,
        "manifest_sha256": manifest_digest(manifest),
        "validation_loss": score,
        "validation_step": validation_step,
        "selection_metric": "held_out_flow_matching_loss_not_rollout_success_rate",
        "provenance": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            **getattr(model, "provenance", {}),
        },
    }


def resume_checkpoint(run, cfg, manifest, world_size, resume):
    path = run / "last.pt"
    if not resume:
        if path.exists():
            raise FileExistsError(f"{path} exists; use --resume to continue it")
        return None
    if not path.exists():
        raise FileNotFoundError(f"--resume requires {path}")
    previous = torch.load(path, map_location="cpu", weights_only=False)
    if previous["config"] != cfg:
        raise ValueError("Resume config differs from checkpoint (including optimizer/schedule/world size)")
    if previous["manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("Resume manifest differs: data split or normalization changed")
    if len(previous["rng_by_rank"]) != world_size:
        raise ValueError("Exact resume requires the same world size")
    return previous


def train(
    run,
    cfg,
    manifest,
    model,
    train_dataset,
    val_dataset,
    device,
    rank=0,
    world_size=1,
    resume=False,
    stop_after=None,
):
    """Train one rank. Injected model/datasets also allow a real CPU/DDP regression."""
    run = Path(run)
    expected = cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"] * world_size
    if cfg["batch_size"] != expected or cfg["world_size"] != world_size:
        raise ValueError("batch_size must equal micro_batch_size * accumulation * world_size")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Pi-0 requires nonempty train and held-out validation splits")
    previous = resume_checkpoint(run, cfg, manifest, world_size, resume)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Pi-0 has no trainable adapter parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"]
    )
    step, best, history, train_seconds, wall_seconds = 0, None, [], 0.0, 0.0
    score, validation_step = None, None
    if previous is not None:
        model.load_trainable_state_dict(previous["adapter"])
        optimizer.load_state_dict(previous["optimizer"])
        step, best, history = previous["step"], previous["best_validation_loss"], previous["history"]
        train_seconds, wall_seconds = previous["train_seconds"], previous["wall_seconds"]
        score, validation_step = previous["validation_loss"], previous["validation_step"]
    worker = model
    if world_size > 1:
        worker = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    # DDP ignores frozen parameters when reducing gradients. Some final VLM
    # branch adapters are unused by the suffix loss, so discovery stays enabled.
    seed_all(cfg["seed"] + rank)
    if previous is not None:
        restore_rng(previous["rng_by_rank"][rank], device)
    del previous
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        write_history(run, history)
    if step >= cfg["train_steps"]:
        if rank == 0 and not (run / "final.pt").exists():
            atomic_checkpoint(
                run / "final.pt", export_checkpoint(model, cfg, manifest, step, score, validation_step)
            )
        return history

    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    end_step = cfg["train_steps"] if stop_after is None else min(step + stop_after, cfg["train_steps"])
    indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(cfg["seed"] + 2026))[
        : cfg["validation_samples"]
    ].tolist()
    if not indices:
        raise ValueError("validation_samples must be positive")
    if rank == 0:
        save_json(run / "validation_indices.json", indices)
    loader_kwargs = dict(
        num_workers=cfg["workers"], pin_memory=device.type == "cuda", persistent_workers=cfg["workers"] > 0
    )
    if cfg["workers"] > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(
        train_dataset,
        batch_sampler=StepMicroBatchSampler(
            len(train_dataset),
            cfg["micro_batch_size"],
            cfg["gradient_accumulation_steps"],
            cfg["train_steps"],
            cfg["seed"],
            rank,
            world_size,
            step,
        ),
        generator=torch.Generator().manual_seed(cfg["seed"] + 12345 + rank),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(val_dataset, indices[rank::world_size]),
        # Keep validation noise grouping stable when migrating training microbatches.
        batch_size=cfg.get("validation_micro_batch_size", cfg["micro_batch_size"]),
        shuffle=False,
        generator=torch.Generator().manual_seed(cfg["seed"] + 54321 + rank),
        **loader_kwargs,
    )
    batches = iter(loader)
    window = torch.zeros(3, device=device, dtype=torch.float64)  # loss, gradient norm, steps
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    train_start = wall_start
    while step < end_step:
        worker.train()
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = accumulate_gradients(worker, batches, cfg["gradient_accumulation_steps"], device)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        window += torch.stack((loss, grad_norm.detach().double(), torch.ones((), device=device)))
        final = step == cfg["train_steps"]
        validate_now = step % cfg["validate_every"] == 0 or final or step == end_step
        save_now = step % cfg["save_every"] == 0 or validate_now or final
        if step % cfg["log_every"] != 0 and not save_now:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        interval = torch.tensor(time.perf_counter() - train_start, device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(window)
            dist.all_reduce(interval, op=dist.ReduceOp.MAX)
        train_seconds += interval.item()
        val_seconds = None
        if validate_now:
            begin = time.perf_counter()
            score = validation(model, val_loader, cfg["seed"], device, rank)
            validation_step = step
            val_seconds = time.perf_counter() - begin
        peak = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
            ],
            device=device,
            dtype=torch.float64,
        )
        peaks = [torch.empty_like(peak) for _ in range(world_size)]
        if world_size > 1:
            dist.all_gather(peaks, peak)
        else:
            peaks[0] = peak
        row = {
            "step": step,
            "train_loss": (window[0] / window[2]).item(),
            "lr": lr,
            "grad_norm": (window[1] / window[2]).item(),
            "validation_loss": score if validate_now else None,
            "validation_seconds": val_seconds,
            "samples_seen": step * cfg["batch_size"],
            "train_seconds": train_seconds,
            "wall_seconds": wall_seconds + time.perf_counter() - wall_start,
            "samples_per_second": window[2].item()
            / world_size
            * cfg["batch_size"]
            / max(interval.item(), 1e-9),
        }
        for gpu, values in enumerate(peaks):
            row[f"gpu_{gpu}_peak_allocated_gib"] = values[0].item()
            row[f"gpu_{gpu}_peak_reserved_gib"] = values[1].item()
        if not math.isfinite(row["train_loss"]):
            raise RuntimeError("Nonfinite training loss")
        if rank == 0:
            append_metrics(run, row, history)
        if save_now:
            rng = capture_rng(device)
            all_rng = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng[0] = rng
            if rank == 0:
                export = export_checkpoint(model, cfg, manifest, step, score, validation_step)
                if validate_now and (best is None or score < best):
                    best = score
                    atomic_checkpoint(run / "best.pt", export)
                atomic_checkpoint(
                    run / "last.pt",
                    {
                        **export,
                        "optimizer": cpu_tree(optimizer.state_dict()),
                        "rng_by_rank": all_rng,
                        "best_validation_loss": best,
                        "history": history,
                        "train_seconds": train_seconds,
                        "wall_seconds": row["wall_seconds"],
                    },
                )
                save_json(run / "train_history.json", history)
                if final:
                    atomic_checkpoint(run / "final.pt", export)
            if world_size > 1:
                dist.barrier()
        window.zero_()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_start = time.perf_counter()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after", type=int, help="Checkpoint after N updates without changing the LR schedule"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    world_size, rank, local_rank = (
        int(os.environ.get(key, default))
        for key, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    if world_size != cfg["world_size"] or not torch.cuda.is_available():
        raise RuntimeError("Launch Pi-0 with torchrun on the configured two CUDA GPUs")
    if torch.cuda.device_count() != world_size:
        raise RuntimeError("Mask CUDA_VISIBLE_DEVICES to exactly the allocated CUDA GPUs")
    from roboscope.runtime.gpu import gpu_model_matches

    gpu_model = cfg.get("gpu_model", "4090")
    if any(not gpu_model_matches(torch.cuda.get_device_name(i), gpu_model) for i in range(world_size)):
        raise RuntimeError(f"Pi-0 recipe requires {gpu_model} GPUs")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This Pi-0 recipe requires BF16 support")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        from roboscope.policies.pi0 import Pi0Policy

        seed_all(cfg["seed"])
        model = Pi0Policy(cfg, manifest).to(device)
        if rank == 0:
            save_json(run / "model_summary.json", model.parameter_counts())
        if manifest.get("dataset_format") == "lerobot_v3":
            from roboscope.data.pi0_lerobot import LeRobotSpatialDataset

            dataset = LeRobotSpatialDataset(manifest, cfg["chunk_size"], "train")
            val_dataset = LeRobotSpatialDataset(manifest, cfg["chunk_size"], "val")
        else:
            from roboscope.data.pi0 import Pi0Dataset

            dataset = Pi0Dataset(manifest, cfg["chunk_size"], "train", run / "image_cache")
            val_dataset = Pi0Dataset(manifest, cfg["chunk_size"], "val", run / "image_cache")
        train(
            run,
            cfg,
            manifest,
            model,
            dataset,
            val_dataset,
            device,
            rank,
            world_size,
            args.resume,
            stop_after=args.stop_after,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
