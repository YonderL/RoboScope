"""Historical ACT chunk/replan/TE evaluation protocol."""

import json
import time
from multiprocessing.connection import wait

import numpy as np
import torch

from roboscope.data.libero import digest, save_json
from roboscope.envs.pool import EnvPool
from roboscope.policies.act import BatchedExecution


def observation_batch(items, device="cuda:0"):
    """只对需要重新预测的 episode 拼 batch；chunk 队列未空时不浪费模型调用。"""
    batch = {
        "state": torch.from_numpy(np.stack([obs["state"] for _, obs in items])),
        "task_id": torch.tensor([task for task, _ in items]),
    }
    for key in ("agentview_rgb", "eye_in_hand_rgb"):
        batch[key] = torch.from_numpy(np.stack([obs[key] for _, obs in items])).permute(0, 3, 1, 2)
    return {k: v.to(device) for k, v in batch.items()}


def run_jobs(model, cfg, chunk, jobs, directory, on_episode, parallelism):
    """动态分发 episode；reset 慢时其他进程继续，不等待整个 batch 一起结束。

    子进程并行 step；主进程把已准备好的 observation 合并后只做一次模型前向。
    只在动作真正传回 CPU 时同步 CUDA，去掉旧实现每步额外的 synchronize。
    模型调用按槽位标识，不按 batch 行号缓存历史，防止不同长度 episode 相互串扰。
    """
    if not jobs:
        return {"wall_seconds": 0.0, "episodes": 0, "model_batches": 0, "env_steps": 0}
    begin = time.perf_counter()
    size = min(parallelism, len(jobs))
    pool = EnvPool(cfg, directory, size)
    execution = BatchedExecution(model, chunk, cfg["temporal_coeff"])
    active, cursor, completed, model_batches, env_steps = {}, 0, 0, 0, 0

    def assign(slot):
        nonlocal cursor
        if cursor >= len(jobs):
            active.pop(slot, None)
            return
        job = jobs[cursor]
        cursor += 1
        execution.add(slot, job["mode"])
        active[slot] = {
            "job": job,
            "phase": "reset",
            "steps": 0,
            "calls": 0,
            "inference_seconds": 0.0,
            "clipped": 0,
        }
        pool.connections[slot].send(("reset", (job["task"], job["initial_id"], job.get("video", ""))))

    try:
        for slot in range(size):
            assign(slot)
        while active:
            stepping = [slot for slot, item in active.items() if item["phase"] == "step"]
            # 等待本轮 step 结束，但不等待仍在 hard reset 的槽位。
            ready_messages = list(stepping)
            ready_messages += [
                slot
                for slot, item in active.items()
                if item["phase"] == "reset" and pool.connections[slot].poll()
            ]
            if not ready_messages:
                busy = {pool.connections[slot]: slot for slot in active}
                ready = wait(list(busy), timeout=180)
                if not ready:
                    raise TimeoutError("No simulation worker responded")
                ready_messages = [busy[conn] for conn in ready]
            for slot in ready_messages:
                item = active[slot]
                obs, success = pool.receive(slot)
                if item["phase"] == "step" and (success or item["steps"] >= cfg["rollout_horizon"]):
                    job = item["job"]
                    on_episode(
                        {
                            **job,
                            "success": int(success),
                            "steps": item["steps"],
                            "model_calls": item["calls"],
                            "clipped_steps": item["clipped"],
                            "inference_seconds": item["inference_seconds"],
                            "inference_timing": "amortized_shared_batch",
                            "rollout_seconds": time.perf_counter() - item["begin"],
                        }
                    )
                    completed += 1
                    execution.remove(slot)
                    assign(slot)
                else:
                    if item["phase"] == "reset":
                        item["begin"] = time.perf_counter()
                    item["obs"], item["phase"] = obs, "ready"
            ready_slots = sorted(slot for slot, item in active.items() if item["phase"] == "ready")
            if not ready_slots:
                continue
            query_slots = [slot for slot in ready_slots if execution.needs_prediction(slot)]
            t = time.perf_counter()
            if query_slots:
                batch = observation_batch(
                    [(active[slot]["job"]["task"]["id"], active[slot]["obs"]) for slot in query_slots]
                )
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                    execution.predict(query_slots, batch)
                model_batches += 1
            with torch.inference_mode():
                # 一个 D2H 同步复制全部动作，而不是每个环境单独同步一次。
                actions = (
                    torch.cat([execution.action(slot) for slot in ready_slots], dim=0).float().cpu().numpy()
                )
            inference_time = time.perf_counter() - t
            if not np.isfinite(actions).all():
                raise RuntimeError("Nonfinite predicted action")
            for slot, action in zip(ready_slots, actions):
                item = active[slot]
                if slot in query_slots:
                    item["calls"] += 1
                    item["inference_seconds"] += inference_time / len(query_slots)
                item["clipped"] += int(np.any(np.abs(action) > 1))
                item["steps"] += 1
                item["phase"] = "step"
                pool.connections[slot].send(("step", np.clip(action, -1, 1)))
                env_steps += 1
    finally:
        pool.close()
    return {
        "wall_seconds": time.perf_counter() - begin,
        "episodes": completed,
        "model_batches": model_batches,
        "env_steps": env_steps,
        "parallel_envs": size,
    }


def evaluate_parallel(run, cfg, manifest, chunk, seed, checkpoint, parallelism=8):
    from roboscope.policies.act import TaskACT
    from roboscope.runtime.common import seed_all

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch, train_seconds = ckpt["epoch"], ckpt["train_seconds"]
    target = run / "eval" / f"epoch_{epoch:04d}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists():
        return
    seed_all(cfg["eval_seed"])
    model = TaskACT(cfg, manifest, chunk, initialize_backbone=False).cuda().eval()
    model.load_state_dict(ckpt["model"])
    del ckpt
    raw, records = target / "episodes.jsonl", []
    if raw.exists():
        for line in raw.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
        raw.write_text("".join(json.dumps(r) + "\n" for r in records))
    done = {(r["task_id"], r["mode"], r["initial_state_id"]) for r in records}
    if len(done) != len(records):
        raise RuntimeError("Duplicate saved episode records")
    modes = cfg["modes"] if chunk != 1 else [cfg["modes"][0]]
    jobs = []
    for task in manifest["tasks"]:
        if digest(task["init"]) != task["init_sha256"] or digest(task["bddl"]) != task["bddl_sha256"]:
            raise RuntimeError("Benchmark files changed")
        for mode in modes:
            for initial_id in task["eval_initial_state_ids"]:
                if (task["id"], mode, initial_id) in done:
                    continue
                record_video = initial_id < cfg["video_episodes_per_task"] and (
                    epoch % cfg["video_every"] == 0 or epoch == cfg["epochs"]
                )
                video = (
                    str(target / f"task{task['id']:02d}_{mode}_init{initial_id:03d}.mp4")
                    if record_video
                    else ""
                )
                jobs.append({"task": task, "mode": mode, "initial_id": initial_id, "video": video})

    def record(result):
        result = dict(result)
        task, initial_id = result.pop("task"), result.pop("initial_id")
        row = {
            **result,
            "epoch": epoch,
            "seed": seed,
            "chunk": chunk,
            "task_id": task["id"],
            "task_name": task["name"],
            "initial_state_id": initial_id,
            "eval_seed": cfg["eval_seed"] + task["id"] * 1000 + initial_id,
            "train_seconds": train_seconds,
            "evaluator": "parallel_v1",
        }
        with raw.open("a") as f:
            f.write(json.dumps(row) + "\n")
        records.append(row)
        done.add((task["id"], row["mode"], initial_id))
        print(
            f"eval epoch={epoch} K={chunk} {row['mode']} task={task['id']} init={initial_id} success={row['success']}",
            flush=True,
        )

    timing = run_jobs(model, cfg, chunk, jobs, target, record, parallelism)
    # 每次恢复只记录本次新增 rollout 的耗时，不能把重复启动误当完整评测时间。
    attempts = target / "timing.jsonl"
    with attempts.open("a") as f:
        f.write(json.dumps(timing) + "\n")
    if chunk == 1:
        originals = [r for r in records if r["mode"] == modes[0]]
        for mode in cfg["modes"][1:]:
            for row in originals:
                if (row["task_id"], mode, row["initial_state_id"]) not in done:
                    clone = {**row, "mode": mode, "reused_from_mode": modes[0]}
                    records.append(clone)
                    with raw.open("a") as f:
                        f.write(json.dumps(clone) + "\n")
    summaries = []
    for mode in cfg["modes"]:
        per_task = []
        for task in manifest["tasks"]:
            selected = [r for r in records if r["mode"] == mode and r["task_id"] == task["id"]]
            if len(selected) != cfg["eval_episodes"]:
                raise RuntimeError("Incomplete or duplicate evaluation records")
            per_task.append(float(np.mean([r["success"] for r in selected])))
        summaries.append(
            {
                "epoch": epoch,
                "seed": seed,
                "chunk": chunk,
                "mode": mode,
                "train_seconds": train_seconds,
                "success_rate": float(np.mean(per_task)),
                "per_task_success": per_task,
            }
        )
    save_json(target / "summary.json", summaries)
