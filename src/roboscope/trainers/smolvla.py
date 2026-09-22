"""SmolVLA fine-tuning with native loss, AdamW and cosine schedule presets."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from roboscope.data.sequences import StepBatchSampler
from roboscope.runtime.common import atomic_save, require_4090, save_json, seed_all
from roboscope.trainers.pi0 import capture_rng, cpu_tree, fixed_rng, manifest_digest, restore_rng


def microbatches(batch, size, device):
    for start in range(0, len(batch["state"]), size):
        yield {
            key: value[start : start + size].to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value[start : start + size]
            for key, value in batch.items()
        }


def backward_batch(model, batch, micro_size, device, use_amp=False):
    """Preserve the native valid-action mean even when microbatches have different tails."""
    total = (~batch["action_is_pad"]).sum().item()
    if total == 0:
        raise ValueError("No valid actions in training batch")
    value = 0.0
    for part in microbatches(batch, micro_size, device):
        weight = (~part["action_is_pad"]).sum() / total
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model(part) * weight
        loss.backward()
        value += loss.detach().item()
    return value


def validation(model, loader, cfg, device, use_amp=False):
    total, count = 0.0, 0
    was_training = model.training
    model.eval()
    try:
        with fixed_rng(cfg["seed"] + 98765, device), torch.no_grad():
            for batch in loader:
                for part in microbatches(batch, cfg["micro_batch_size"], device):
                    weight = (~part["action_is_pad"]).sum().item()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                        loss = model(part)
                    total += loss.item() * weight
                    count += weight
        if count == 0 or not math.isfinite(total):
            raise RuntimeError("Empty or nonfinite SmolVLA validation")
        return total / count
    finally:
        model.train(was_training)


def train(run, cfg, manifest, model, dataset, val_dataset, device, resume=False):
    """Injectable datasets/model allow optimizer and exact-restart CPU regression tests."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    native = model.policy.config
    optimizer_preset = native.get_optimizer_preset()
    optimizer = optimizer_preset.build(model.parameters())
    scheduler = native.get_scheduler_preset().build(optimizer, cfg["train_steps"])
    use_amp = native.use_amp and device.type == "cuda"
    identity = manifest_digest(manifest)
    step, best, history, score = 0, None, [], None
    seed_all(cfg["seed"])
    if resume:
        saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["manifest_sha256"] != identity:
            raise ValueError("SmolVLA resume config or manifest changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], device)
        step, best, history, score = saved["step"], saved["best"], saved["history"], saved["validation_loss"]
        del saved
    elif (run / "last.pt").exists():
        raise FileExistsError("Existing last.pt; use --resume")

    def export():
        return {
            "format": "roboscope.smolvla.v1",
            "model": cpu_tree(model.state_dict()),
            "step": step,
            "config": cfg,
            "manifest_sha256": identity,
            "validation_loss": score,
            "selection_metric": "held_out_flow_matching_loss_not_rollout_success_rate",
        }

    save_json(run / "train_history.json", history)
    if step >= cfg["train_steps"]:
        if not (run / "final.pt").exists():
            atomic_save(run / "final.pt", export())
        return history
    indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(2026))[
        : cfg["validation_samples"]
    ].tolist()
    save_json(run / "validation_indices.json", indices)
    kwargs = dict(
        num_workers=cfg["workers"], pin_memory=device.type == "cuda", persistent_workers=cfg["workers"] > 0
    )
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
        Subset(val_dataset, indices),
        batch_size=cfg["micro_batch_size"],
        generator=torch.Generator().manual_seed(54321),
        **kwargs,
    )
    for batch in loader:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = backward_batch(model, batch, cfg["micro_batch_size"], device, use_amp)
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optimizer_preset.grad_clip_norm, error_if_nonfinite=True
        )
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite SmolVLA training loss")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        step += 1
        final = step == cfg["train_steps"]
        validate_now = step % cfg["validate_every"] == 0 or final
        save_now = step % cfg["save_every"] == 0 or validate_now or final
        if validate_now:
            score = validation(model, val_loader, cfg, device, use_amp)
        if step % cfg["log_every"] == 0 or save_now:
            row = {
                "step": step,
                "train_loss": loss,
                "lr": lr,
                "grad_norm": float(grad),
                "validation_loss": score if validate_now else None,
                "samples_seen": step * cfg["batch_size"],
            }
            history.append(row)
            save_json(run / "train_history.json", history)
            print(json.dumps(row), flush=True)
        if save_now:
            payload = export()
            if validate_now and (best is None or score < best):
                best = score
                atomic_save(run / "best.pt", payload)
            atomic_save(
                run / "last.pt",
                {
                    **payload,
                    "optimizer": cpu_tree(optimizer.state_dict()),
                    "scheduler": scheduler.state_dict(),
                    "rng": capture_rng(device),
                    "best": best,
                    "history": history,
                },
            )
            if final:
                atomic_save(run / "final.pt", payload)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    require_4090()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    seed_all(cfg["seed"])
    from roboscope.data.smolvla import SmolVLADataset
    from roboscope.policies.smolvla import SmolVLAPolicy

    model = SmolVLAPolicy(cfg, manifest, initialize_pretrained=not args.resume).cuda()
    model.policy.config.save_pretrained(args.run / "backend_config")
    save_json(
        args.run / "model_summary.json",
        {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "training_amp": model.policy.config.use_amp,
        },
    )
    datasets = [
        SmolVLADataset(manifest, cfg["chunk_size"], split, args.run / "image_cache")
        for split in ("train", "val")
    ]
    train(args.run, cfg, manifest, model, *datasets, torch.device("cuda"), args.resume)


if __name__ == "__main__":
    main()
