"""Shared fixed-state ACT/DP/SmolVLA evaluation with per-episode histories and RNG."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from roboscope.data.sequences import SequenceDataset
from roboscope.envs.pool import EnvPool
from roboscope.policies.act import TaskACT
from roboscope.policies.diffusion import TaskDiffusionPolicy
from roboscope.runtime.common import CAMERAS, digest, load_config, require_4090, save_json, seed_all, variants


def batch_observations(slots, kind, device="cuda"):
    result = {"task_id": torch.tensor([s["task"]["id"] for s in slots], device=device)}
    for key in ("state", *CAMERAS):
        if kind == "dp":
            arr = np.stack([np.stack([o[key] for o in s["history"]]) for s in slots])
        else:
            arr = np.stack([s["history"][-1][key] for s in slots])
        tensor = torch.from_numpy(arr)
        if key in CAMERAS:
            tensor = tensor.permute(0, 1, 4, 2, 3) if kind == "dp" else tensor.permute(0, 3, 1, 2)
        result[key] = tensor.to(device)
    return result


def load_policy(run, cfg, manifest, env_cfg, variant):
    if variant["model"] == "rlt":
        from roboscope.policies.smolvla_rlt import load_policy as load_rlt

        path = run / (cfg["evaluation_checkpoint"] + ".pt")
        model, step = load_rlt(path, manifest, cfg, "cuda")
    elif variant["model"] == "smolvla":
        from roboscope.policies.smolvla import SmolVLAPolicy

        path = run / (cfg["evaluation_checkpoint"] + ".pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = SmolVLAPolicy(ckpt["config"], manifest, initialize_pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        step = ckpt["step"]
    elif variant["model"] == "dp":
        path = run / (cfg["evaluation_checkpoint"] + ".pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = TaskDiffusionPolicy(cfg, manifest)
        model.load_state_dict(ckpt["model"])
        step = ckpt["step"]
    else:
        path = Path(cfg["act_checkpoint"])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = TaskACT(cfg["act_model_config"], manifest, 8, initialize_backbone=False)
        model.load_state_dict(ckpt["model"])
        step = ckpt["global_step"]
    return model.cuda().eval(), {
        "checkpoint": str(path),
        "checkpoint_sha256": digest(path),
        "gradient_step": step,
    }


def latency(model, variant, cfg, manifest, cache):
    """统一 batch=1；输入已在 GPU，计时包含预处理、完整采样及动作 D2H。

    不把 8 环境 batch 的摊销时间冒充单机器人 latency。不含图像获取/网络/仿真。
    每次是完整 chunk 生成，不除以 Ta；热身后同步测 wall-clock p50/p95。
    """
    ds = SequenceDataset(manifest, "val", cache)
    sample = ds[0]
    batch = {k: v.unsqueeze(0).cuda() for k, v in sample.items() if k not in ("action", "action_is_pad")}
    if variant["model"] != "dp":
        for key in ("state", *CAMERAS):
            batch[key] = batch[key][:, -1]
    times = []
    with torch.inference_mode():
        for i in range(cfg["latency_warmup"] + cfg["latency_repeats"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                prediction = (
                    model.predict(batch, variant["ddim_steps"])
                    if variant["model"] == "dp"
                    else model.predict(batch)
                )
            prediction.float().cpu()
            torch.cuda.synchronize()
            if i >= cfg["latency_warmup"]:
                times.append((time.perf_counter() - start) * 1000)
    return {
        "batch_size": 1,
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "samples_ms": times,
        "scope": "GPU-resident raw observation -> preprocess/CNN/full chunk generation -> CPU actions; no environment/network",
        "gpu": torch.cuda.get_device_name(0),
    }


def rollout(model, cfg, env_cfg, jobs, variant, target, callback):
    begin = time.perf_counter()
    if not jobs:
        return {"wall_seconds": 0.0, "episodes": 0}
    pool = EnvPool(env_cfg, target, min(cfg["eval_envs"], len(jobs)))
    active = {}
    cursor = 0
    completed = 0
    batches = 0
    env_steps = 0

    def assign(slot):
        nonlocal cursor
        if cursor == len(jobs):
            active.pop(slot, None)
            return
        task, initial_id = jobs[cursor]
        cursor += 1
        seed = cfg["eval_seed"] + task["id"] * 1000 + initial_id
        video = (
            str(target / f"task{task['id']:02d}_init{initial_id:03d}.mp4")
            if initial_id < cfg["video_episodes_per_task"]
            else ""
        )
        active[slot] = {
            "task": task,
            "initial_id": initial_id,
            "steps": 0,
            "calls": 0,
            "phase": "reset",
            "history": deque(maxlen=2),
            "queue": deque(),
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "states": [],
            "eef": [],
            "raw_actions": [],
            "actions": [],
            "query_steps": [],
            "begin": time.perf_counter(),
        }
        pool.connections[slot].send(("reset", (task, initial_id, video)))

    try:
        for slot in range(len(pool.connections)):
            assign(slot)
        while active:
            # 同 ACT：step 之间有同步屏障。仿真并行，策略每次批量同步预测。
            for slot in list(active):
                item = active[slot]
                obs, success = pool.receive(slot)
                item["history"].append(obs)
                if item["phase"] == "reset":
                    item["history"].append(obs)
                if item["phase"] == "step" and (success or item["steps"] >= cfg["rollout_horizon"]):
                    item["states"].append(obs["state"])
                    item["eef"].append(obs["eef_pos"])
                    actions = np.asarray(item["actions"])
                    eef = np.asarray(item["eef"])
                    trace = target / f"task{item['task']['id']:02d}_init{item['initial_id']:03d}.npz"
                    np.savez_compressed(
                        trace,
                        states=np.asarray(item["states"]),
                        eef_pos=eef,
                        actions=actions,
                        raw_actions=np.asarray(item["raw_actions"]),
                        query_steps=np.asarray(item["query_steps"]),
                    )
                    delta = np.diff(actions, axis=0)
                    result = {
                        "task_id": item["task"]["id"],
                        "task_name": item["task"]["name"],
                        "initial_state_id": item["initial_id"],
                        "success": int(success),
                        "steps": item["steps"],
                        "model_calls": item["calls"],
                        "query_fraction": item["calls"] / item["steps"],
                        "nominal_replanning_hz": 20 * item["calls"] / item["steps"],
                        "eef_path_length": float(np.linalg.norm(np.diff(eef, axis=0), axis=-1).sum()),
                        "action_delta_rms": float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.0,
                        "translation_action_delta_rms": float(np.sqrt(np.mean(delta[:, :3] ** 2)))
                        if len(delta)
                        else 0.0,
                        "rotation_action_delta_rms": float(np.sqrt(np.mean(delta[:, 3:6] ** 2)))
                        if len(delta)
                        else 0.0,
                        "gripper_switches": int(np.count_nonzero(np.diff(actions[:, -1] > 0))),
                        "clipped_step_fraction": float(
                            np.mean(np.any(np.abs(item["raw_actions"]) > 1, axis=-1))
                        ),
                        "rollout_seconds": time.perf_counter() - item["begin"],
                        "trace": str(trace),
                        "eval_seed": cfg["eval_seed"] + item["task"]["id"] * 1000 + item["initial_id"],
                    }
                    callback(result)
                    completed += 1
                    assign(slot)
                # 新 episode 的 reset 在下一轮接收，不使用旧 episode 的 observation。
                else:
                    item["phase"] = "ready"
            ready = [slot for slot in sorted(active) if active[slot]["phase"] == "ready"]
            query = [slot for slot in ready if not active[slot]["queue"]]
            if query:
                items = [active[s] for s in query]
                batch = batch_observations(items, variant["model"])
                if variant["model"] == "rlt":
                    batch["remaining_steps"] = torch.tensor(
                        [cfg["rollout_horizon"] - item["steps"] for item in items], device="cuda"
                    )
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                    if variant["model"] == "dp":
                        noise = torch.stack(
                            [torch.randn(16, 7, device="cuda", generator=s["generator"]) for s in items]
                        )
                        predicted = model.predict(batch, variant["ddim_steps"], noise=noise)
                    elif variant["model"] in ("smolvla", "rlt"):
                        noise = torch.stack(
                            [torch.randn(50, 32, device="cuda", generator=s["generator"]) for s in items]
                        )
                        predicted = model.predict(batch, noise=noise)
                    else:
                        predicted = model.predict(batch)
                chunks = predicted[:, : variant["ta"]].float().cpu().numpy()
                if not np.isfinite(chunks).all():
                    raise RuntimeError("Nonfinite action")
                for item, chunk in zip(items, chunks, strict=True):
                    item["queue"].extend(chunk.copy())
                    item["calls"] += 1
                    item["query_steps"].append(item["steps"])
                batches += 1
            for slot in ready:
                item = active[slot]
                raw = item["queue"].popleft()
                action = np.clip(raw, -1, 1)
                item["states"].append(item["history"][-1]["state"])
                item["eef"].append(item["history"][-1]["eef_pos"])
                item["raw_actions"].append(raw)
                item["actions"].append(action)
                item["steps"] += 1
                item["phase"] = "step"
                env_steps += 1
                pool.connections[slot].send(("step", action))
    finally:
        pool.close()
    return {
        "wall_seconds": time.perf_counter() - begin,
        "episodes": completed,
        "env_steps": env_steps,
        "model_batches": batches,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--shard", type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    cfg = load_config(a.run / "config.json")
    require_4090()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    seed_all(cfg["seed"])
    manifest = json.loads((a.run / "manifest.json").read_text())
    env_cfg = json.loads((a.run / "env_config.json").read_text())
    variant = next(v for v in variants(cfg) if v["name"] == a.variant)
    target = a.run / "eval" / variant["name"] / f"shard{a.shard}"
    target.mkdir(parents=True, exist_ok=True)
    model, identity = load_policy(a.run, cfg, manifest, env_cfg, variant)
    metadata = {
        "config": cfg,
        "variant": variant,
        "shard": a.shard,
        **identity,
        "task_ids": [t["id"] for t in manifest["tasks"] if t["id"] % 2 == a.shard],
    }
    if (target / "config.json").exists() and json.loads((target / "config.json").read_text()) != metadata:
        raise RuntimeError("Evaluation identity changed; refusing mixed checkpoints")
    save_json(target / "config.json", metadata)
    records = []
    raw = target / "episodes.jsonl"
    if raw.exists():
        for line in raw.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
        raw.write_text("".join(json.dumps(r) + "\n" for r in records))
    done = {(r["task_id"], r["initial_state_id"]) for r in records}
    if len(done) != len(records):
        raise RuntimeError("Duplicate episodes")
    jobs = []
    for task in manifest["tasks"]:
        if task["id"] % 2 != a.shard:
            continue
        if digest(task["init"]) != task["init_sha256"] or digest(task["bddl"]) != task["bddl_sha256"]:
            raise RuntimeError("Changed initial states/BDDL")
        jobs += [(task, i) for i in task["eval_initial_state_ids"] if (task["id"], i) not in done]
    if not (target / "latency.json").exists():
        save_json(target / "latency.json", latency(model, variant, cfg, manifest, a.run / "image_cache"))

    def record(row):
        row = {**row, "variant": variant["name"], "checkpoint_sha256": identity["checkpoint_sha256"]}
        with raw.open("a") as f:
            f.write(json.dumps(row) + "\n")
        records.append(row)
        print(
            json.dumps({k: row[k] for k in ["variant", "task_id", "initial_state_id", "success", "steps"]}),
            flush=True,
        )

    timing = rollout(model, cfg, env_cfg, jobs, variant, target, record)
    with (target / "timing.jsonl").open("a") as f:
        f.write(json.dumps(timing) + "\n")
    expected = len(metadata["task_ids"]) * cfg["eval_episodes"]
    if len(records) != expected:
        raise RuntimeError("Incomplete evaluation")
    save_json(
        target / "complete.json",
        {"episodes": len(records), "checkpoint_sha256": identity["checkpoint_sha256"]},
    )


if __name__ == "__main__":
    main()
