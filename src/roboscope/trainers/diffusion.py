"""One suite-conditioned DP, one seed. Run through launch.py, never on an unmasked GPU."""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from roboscope.data.prefetch import device_batches
from roboscope.data.sequences import SequenceDataset, StepBatchSampler
from roboscope.policies.diffusion import TaskDiffusionPolicy, update_ema
from roboscope.runtime.common import atomic_save, load_config, require_4090, save_json, seed_all


def validation(model, loader, cfg):
    model.eval()
    total = torch.zeros(2, device="cuda", dtype=torch.float64)
    # 固定噪声和验证样本，使 checkpoint 的分数可比较；不消耗训练 RNG。
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(cfg["seed"] + 98765)
        torch.cuda.manual_seed(cfg["seed"] + 98765)
        with torch.no_grad():
            for batch in device_batches(loader):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                    loss = model(batch)
                weight = (~batch["action_is_pad"]).sum() if cfg["mask_padding_loss"] else len(batch["state"])
                total[0] += loss.double() * weight
                total[1] += weight
    return (total[0] / total[1]).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    cfg = load_config(run / "config.json")
    require_4090()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    seed_all(cfg["seed"])
    manifest = json.loads((run / "manifest.json").read_text())
    resume = (
        torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if (run / "last.pt").exists()
        else None
    )
    if resume and resume["step"] >= cfg["train_steps"]:
        if not (run / "final.pt").exists():
            atomic_save(
                run / "final.pt",
                {
                    "model": resume["ema"],
                    "step": resume["step"],
                    "config": cfg,
                    "validation_mse": resume["validation_mse"],
                },
            )
        return
    model = TaskDiffusionPolicy(cfg, manifest).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.95, 0.999), weight_decay=cfg["weight_decay"]
    )
    step, best, history, elapsed = 0, float("inf"), [], 0.0
    if resume:
        model.load_state_dict(resume["model"])
        ema.load_state_dict(resume["ema"])
        optimizer.load_state_dict(resume["optimizer"])
        step, best, history, elapsed = (
            resume["step"],
            resume["best"],
            resume["history"],
            resume["train_seconds"],
        )
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        del resume
    dataset = SequenceDataset(manifest, "train", run / "image_cache")
    val = SequenceDataset(manifest, "val", run / "image_cache")
    indices = torch.randperm(len(val), generator=torch.Generator().manual_seed(2026))[
        : cfg["validation_samples"]
    ].tolist()
    kwargs = dict(num_workers=cfg["workers"], pin_memory=True, persistent_workers=cfg["workers"] > 0)
    if cfg["workers"]:
        kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(
        dataset,
        batch_sampler=StepBatchSampler(
            len(dataset), cfg["batch_size"], cfg["train_steps"], cfg["seed"], step
        ),
        generator=torch.Generator().manual_seed(12345),
        **kwargs,
    )
    val_loader = DataLoader(
        Subset(val, indices),
        batch_size=cfg["batch_size"],
        shuffle=False,
        generator=torch.Generator().manual_seed(54321),
        **kwargs,
    )
    save_json(run / "validation_indices.json", indices)
    totals = torch.zeros(2, device="cuda", dtype=torch.float64)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for batch in device_batches(loader):
        model.train()
        # global batch 不变；microbatch 用于显存不足时梯度累积。
        optimizer.zero_grad(set_to_none=True)
        micro = cfg.get("micro_batch_size", cfg["batch_size"])
        loss_sum = torch.zeros((), device="cuda")
        valid_total = (~batch["action_is_pad"]).sum() if cfg["mask_padding_loss"] else len(batch["state"])
        for start in range(0, len(batch["state"]), micro):
            part = {k: v[start : start + micro] for k, v in batch.items()}
            weight = (
                (~part["action_is_pad"]).sum() if cfg["mask_padding_loss"] else len(part["state"])
            ) / valid_total
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                loss = model(part) * weight
            loss.backward()
            loss_sum += loss.detach()
        fraction = (
            min(1.0, (step + 1) / cfg["warmup_steps"])
            if step < cfg["warmup_steps"]
            else 0.5
            * (
                1
                + math.cos(
                    math.pi * (step - cfg["warmup_steps"]) / max(1, cfg["train_steps"] - cfg["warmup_steps"])
                )
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * fraction
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        update_ema(ema, model, step, cfg)
        totals[0] += loss_sum.double()
        totals[1] += 1
        if step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_noise_mse": (totals[0] / totals[1]).item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                ),
                flush=True,
            )
        if step % cfg["validate_every"] == 0 or step == cfg["train_steps"]:
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - begin
            val_begin = time.perf_counter()
            score = validation(ema, val_loader, cfg)
            if not math.isfinite(score):
                raise RuntimeError("Nonfinite validation")
            row = {
                "step": step,
                "train_noise_mse": (totals[0] / totals[1]).item(),
                "val_noise_mse_ema": score,
                "train_seconds": elapsed,
                "validation_seconds": time.perf_counter() - val_begin,
                "samples_seen": step * cfg["batch_size"],
            }
            history.append(row)
            save_json(run / "train_history.json", history)
            export = {"model": ema.state_dict(), "step": step, "config": cfg, "validation_mse": score}
            if score < best:
                best = score
                atomic_save(run / "best.pt", export)
            # best 不代表 rollout 最优。测试成功率不参与 checkpoint 选择。
            atomic_save(
                run / "last.pt",
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best": best,
                    "validation_mse": score,
                    "history": history,
                    "train_seconds": elapsed,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                },
            )
            if step == cfg["train_steps"]:
                atomic_save(run / "final.pt", export)
            print(json.dumps(row), flush=True)
            totals.zero_()
            torch.cuda.synchronize()
            begin = time.perf_counter()


if __name__ == "__main__":
    main()
