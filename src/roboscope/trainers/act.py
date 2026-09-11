"""ACT training worker; executed as a module by the workflow."""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roboscope.data.libero import FrameDataset, save_json
from roboscope.data.prefetch import device_batches
from roboscope.policies.act import TaskACT


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def evaluate_checkpoint(run, checkpoint):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "roboscope.trainers.act",
            "eval",
            "--run",
            str(run),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
    )


def train(run, cfg, manifest, chunk, seed):
    """一个进程训练一个 K/seed，不做 DDP；两个 4090 各跑独立实验。

    每 epoch 完整遍历训练帧，保存模型、优化器、随机数状态和历史，实现 epoch
    边界恢复。评测使用独立子进程，不改变训练的随机数状态。
    train_seconds 只累计训练循环（包含取数据），不含验证、rollout 和保存耗时。
    """
    runtime = json.loads((run / "run.json").read_text()).get("runtime", {})
    cache_path = run.parent / "image_cache" if runtime.get("image_cache", False) else None
    last = run / "last.pt"
    previous = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    seed_all(seed)
    model = TaskACT(cfg, manifest, chunk, initialize_backbone=previous is None).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history, elapsed, global_step, start = [], 0.0, 0, 1
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        history, elapsed = previous["history"], previous["train_seconds"]
        global_step, start = previous["global_step"], previous["epoch"] + 1
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state(previous["cuda_rng"], device=0)
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        # Recover evaluation interrupted after a completed training epoch.
        if previous["epoch"] % cfg["eval_every"] == 0 or previous["epoch"] == cfg["epochs"]:
            evaluate_checkpoint(run, last)
        del previous
    save_json(run / "train_history.json", history)
    dataset = FrameDataset(manifest, chunk, "train", image_cache=cache_path)
    validation = FrameDataset(manifest, chunk, "val", image_cache=cache_path)
    generator = torch.Generator()
    kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        persistent_workers=cfg["workers"] > 0,
    )
    if cfg["workers"] > 0:
        kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, shuffle=True, generator=generator, **kwargs)
    val_generator = torch.Generator().manual_seed(seed + 700000)
    val_loader = DataLoader(validation, shuffle=False, generator=val_generator, **kwargs)
    for epoch in range(start, cfg["epochs"] + 1):
        # Sampling population and number of optimizer steps are independent of chunk size.
        generator.manual_seed(seed * 100000 + epoch)
        model.train()
        totals = {
            k: torch.zeros((), device="cuda", dtype=torch.float64) for k in ("loss", "l1_loss", "kld_loss")
        }
        count = 0
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for batch in device_batches(loader, enabled=runtime.get("prefetch_cuda", True)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                loss, metrics = model.loss_tensors(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            n = len(batch["state"])
            count += n
            for key, value in {"loss": loss.detach(), **metrics}.items():
                totals[key] += value.double() * n
            global_step += 1
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - begin
        # Validation reports deterministic z=0 prediction error, not posterior VAE loss.
        model.eval()
        val_begin = time.perf_counter()
        errors = torch.zeros((), device="cuda", dtype=torch.float64)
        valid = torch.zeros((), device="cuda", dtype=torch.int64)
        with torch.no_grad():
            for batch in device_batches(val_loader, enabled=runtime.get("prefetch_cuda", True)):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                    predicted = model.predict(batch)
                mask = (~batch["action_is_pad"]).unsqueeze(-1)
                errors += (((predicted - batch["action"]).abs() / model.action_std) * mask).sum().double()
                valid += mask.sum() * 7
        torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_seconds": elapsed,
            "val_seconds": time.perf_counter() - val_begin,
            "val_l1_z0": (errors / valid).item(),
            **{k: v.item() / count for k, v in totals.items()},
        }
        if not all(np.isfinite(row[k]) for k in ("loss", "val_l1_z0")):
            raise RuntimeError("Nonfinite epoch metrics")
        history.append(row)
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "train_seconds": elapsed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        atomic_checkpoint(last, payload)
        save_json(run / "train_history.json", history)
        print(json.dumps(row), flush=True)
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            if cfg["keep_every_checkpoint"]:
                atomic_checkpoint(run / f"epoch_{epoch:04d}.pt", payload)
            torch.cuda.empty_cache()
            evaluate_checkpoint(run, last)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    spec = json.loads((run / "run.json").read_text())
    cfg = spec["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one CUDA device")
    if "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Refusing computation on a non-4090 GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((run.parent / "manifest.json").read_text())
    if args.command == "train":
        train(run, cfg, manifest, spec["chunk"], spec["seed"])
    else:
        from roboscope.evaluation.act import evaluate_parallel

        evaluate_parallel(
            run,
            cfg,
            manifest,
            spec["chunk"],
            spec["seed"],
            args.checkpoint or run / "last.pt",
            parallelism=spec.get("runtime", {}).get("eval_envs", 8),
        )


if __name__ == "__main__":
    main()
