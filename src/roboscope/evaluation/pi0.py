"""Fixed-state Pi0 rollouts, durable episode records, and strict shard aggregation.

One UUID-masked GPU evaluates one task shard. Keep raw simulator observations:
the ACT/DP pool's compact state omits the end-effector orientation Pi0 requires.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback
from collections import deque
from pathlib import Path


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def append_record(path, row):
    """An interrupted run may only lose its currently executing episode."""
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records, good_bytes, repaired = [], 0, False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair_tail or index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt episode record at {path}:{index + 1}") from None
            with path.open("r+b") as handle:
                handle.truncate(good_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            repaired = True
            break
        if not isinstance(row, dict):
            raise ValueError(f"Episode record is not an object at {path}:{index + 1}")
        records.append(row)
        good_bytes += len(line)
    # Complete JSON without a newline is also recoverable; preserve its record.
    if (
        not repaired
        and records
        and lines
        and not lines[-1].endswith(b"\n")
        and good_bytes == path.stat().st_size
    ):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return records


def wilson_interval(successes, episodes):
    """95% binomial Wilson interval; conditional on this fixed task/init suite."""
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError("Wilson interval requires 0 <= successes <= positive episodes")
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes**2)) / denominator
    return [
        0.0 if successes == 0 else max(0.0, center - radius),
        1.0 if successes == episodes else min(1.0, center + radius),
    ]


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_records(records, task_ids, episodes, checkpoint_sha256, horizon=220):
    """Do not turn exceptions, missing trials, or duplicate trials into failures."""
    expected = {(task, initial) for task in task_ids for initial in range(episodes)}
    seen = set()
    for row in records:
        key = (row["task_id"], row["initial_state_id"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected evaluation episode")
        seen.add(key)
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("Mixed checkpoint identity in episode records")
        if type(row.get("success")) is not int or row["success"] not in (0, 1):
            raise ValueError("Episode success must be an observed binary outcome")
        if type(row.get("steps")) is not int or not 1 <= row["steps"] <= horizon:
            raise ValueError("Episode steps violate evaluation horizon")
        if row.get("error"):
            raise ValueError("Simulator errors cannot be counted as policy failures")
    if seen != expected:
        raise ValueError(f"Incomplete evaluation: expected {len(expected)} episodes, found {len(seen)}")
    per_task = []
    for task in task_ids:
        selected = [row for row in records if row["task_id"] == task]
        successes = sum(row["success"] for row in selected)
        per_task.append(
            {
                "task_id": task,
                "task_name": selected[0]["task_name"],
                "episodes": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected),
                "success_rate_wilson_95": wilson_interval(successes, len(selected)),
                "mean_steps": _mean([row["steps"] for row in selected]),
                "mean_success_steps": _mean([row["steps"] for row in selected if row["success"]]),
            }
        )
    successes = sum(row["success"] for row in records)
    samples = [value for row in records for value in row.get("inference_samples_ms", [])]
    steps = sum(row["steps"] for row in records)
    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else None,
        "macro_task_success_rate": _mean([row["success_rate"] for row in per_task]),
        "success_rate_wilson_95": wilson_interval(successes, len(records)) if records else None,
        "ci_scope": "Descriptive binomial interval over fixed task/init trials; not cross-seed uncertainty",
        "per_task": per_task,
        "mean_steps": _mean([row["steps"] for row in records]),
        "mean_success_steps": _mean([row["steps"] for row in records if row["success"]]),
        "total_control_steps": steps,
        "sum_episode_seconds": sum(row.get("rollout_seconds", 0) for row in records),
        "model_calls": sum(row.get("model_calls", 0) for row in records),
        "clipped_step_fraction": sum(row.get("clipped_steps", 0) for row in records) / steps
        if steps
        else None,
        "rollout_inference_batch_size": 1,
        "rollout_inference_mean_ms": _mean(samples),
        "rollout_inference_p50_ms": _percentile(samples, 0.5),
        "rollout_inference_p95_ms": _percentile(samples, 0.95),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)


def aggregate_evaluation(output):
    """Called by the workflow only after both GPU workers exit successfully."""
    output = Path(output)
    identity, records = None, []
    for shard in range(2):
        folder = output / f"shard_{shard}"
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata.pop("shard") != shard:
            raise ValueError("Shard identity mismatch")
        task_ids = metadata.pop("shard_task_ids")
        complete = json.loads((folder / "complete.json").read_text())
        if identity is not None and identity != metadata:
            raise ValueError("Mixed evaluation identities across shards")
        identity = metadata
        rows = read_records(folder / "episodes.jsonl")
        summarize_records(
            rows,
            task_ids,
            metadata["episodes_per_task"],
            metadata["checkpoint_sha256"],
            metadata["protocol"]["rollout_horizon"],
        )
        if complete != {"episodes": len(rows), "checkpoint_sha256": metadata["checkpoint_sha256"]}:
            raise ValueError("Shard completion identity mismatch")
        records.extend(rows)
    summary = summarize_records(
        records,
        identity["task_ids"],
        identity["episodes_per_task"],
        identity["checkpoint_sha256"],
        identity["protocol"]["rollout_horizon"],
    )
    summary.update(protocol=identity["protocol"], smoke=identity["smoke"], checkpoint=identity["checkpoint"])
    _save_json(output / "summary.json", summary)
    _save_json(output / "evaluation_identity.json", identity)
    _write_csv(
        output / "episodes.csv", sorted(records, key=lambda row: (row["task_id"], row["initial_state_id"]))
    )
    _write_csv(output / "task_metrics.csv", summary["per_task"])
    return summary


def _predict(model, obs, task, cfg, generator):
    import numpy as np
    import torch

    from roboscope.data.pi0 import observation_batch

    start = time.perf_counter()
    batch = observation_batch(obs, task, "cuda")
    noise = torch.randn(
        (1, cfg["chunk_size"], cfg["max_action_dim"]), generator=generator, device="cuda", dtype=torch.float32
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        action = model.predict(batch, noise=noise).float().cpu().numpy()
    elapsed = (time.perf_counter() - start) * 1000
    if action.shape != (1, cfg["chunk_size"], 7) or not np.isfinite(action).all():
        raise RuntimeError(f"Pi0 returned invalid raw OSC action chunk: {action.shape}")
    return action[0], elapsed


def _latency(model, obs, task, cfg):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(cfg["eval_seed"] + 999999)
    samples = []
    warmup, repeats = cfg.get("latency_warmup", 5), cfg.get("latency_repeats", 30)
    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid latency sample counts")
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        _, elapsed = _predict(model, obs, task, cfg, generator)
        if index >= warmup:
            samples.append(elapsed)
    return {
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "samples_ms": samples,
        "mean_ms": _mean(samples),
        "p50_ms": _percentile(samples, 0.5),
        "p95_ms": _percentile(samples, 0.95),
        "gpu": torch.cuda.get_device_name(0),
        "scope": "CPU raw observation -> preprocessing/H2D/full flow chunk -> CPU actions; no simulator/video",
        "num_inference_steps": cfg["num_inference_steps"],
        "chunk_size": cfg["chunk_size"],
    }


def run_episode(model, env, task, initial_states, initial_id, cfg, video_path=None, latency_path=None):
    import numpy as np
    import torch

    from roboscope.envs.libero import video_frame
    from roboscope.runtime.common import seed_all

    episode_seed = cfg["eval_seed"] + task["id"] * 1000 + initial_id
    seed_all(episode_seed)
    env.seed(episode_seed)
    begin = time.perf_counter()
    env.reset()
    obs = env.set_init_state(np.asarray(initial_states[initial_id]))
    # Same settling command as openpi's LIBERO evaluator: keep the gripper open.
    for _ in range(cfg["settle_steps"]):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    if latency_path is not None and not Path(latency_path).exists():
        _save_json(latency_path, _latency(model, obs, task, cfg))
    rollout_begin = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(episode_seed)
    queue, samples = deque(), []
    writer = None
    steps, clipped, success = 0, 0, False
    try:
        if video_path:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(video_path), fps=20, codec="libx264", macro_block_size=1, ffmpeg_params=["-threads", "1"]
            )
            writer.append_data(video_frame(obs))
        for _ in range(cfg["rollout_horizon"]):
            if not queue:
                actions, elapsed = _predict(model, obs, task, cfg, generator)
                queue.extend(actions[: cfg["action_horizon"]])
                samples.append(elapsed)
            raw = queue.popleft()
            clipped += int(np.any(np.abs(raw) > 1))
            obs, _, _, _ = env.step(np.clip(raw, -1, 1).tolist())
            steps += 1
            success = bool(env.check_success())
            if writer is not None:
                writer.append_data(video_frame(obs))
            if success:
                break
    finally:
        if writer is not None:
            writer.close()
    return {
        "task_id": task["id"],
        "task_name": task["name"],
        "language": task["language"],
        "initial_state_id": initial_id,
        "eval_seed": episode_seed,
        "success": int(success),
        "steps": steps,
        "model_calls": len(samples),
        "clipped_steps": clipped,
        "inference_samples_ms": samples,
        "inference_mean_ms": _mean(samples),
        "inference_p50_ms": _percentile(samples, 0.5),
        "inference_p95_ms": _percentile(samples, 0.95),
        "rollout_seconds": time.perf_counter() - rollout_begin,
        "episode_seconds_including_reset": time.perf_counter() - begin,
        "video": str(video_path) if video_path else None,
    }


def evaluate_shard(run, output, checkpoint="final", shard=0, episodes=50, max_tasks=None):
    import numpy as np
    import torch

    from roboscope.envs.libero import setup_libero
    from roboscope.policies.pi0 import Pi0Policy
    from roboscope.runtime.common import require_4090, seed_all

    if shard not in (0, 1) or episodes < 1 or (max_tasks is not None and not 1 <= max_tasks <= 10):
        raise ValueError("Invalid shard, episode count, or maximum task count")
    if checkpoint not in ("final", "best"):
        raise ValueError("Evaluation checkpoint must be final or validation-loss best")
    require_4090()
    run, output = Path(run).resolve(), Path(output).resolve()
    cfg = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    tasks = sorted(manifest["tasks"], key=lambda task: task["id"])
    if len(tasks) != 10 or len({task["id"] for task in tasks}) != 10:
        raise ValueError("Pi0 evaluator requires all ten Spatial tasks in the training manifest")
    tasks = tasks[:max_tasks] if max_tasks is not None else tasks
    if not 1 <= cfg["action_horizon"] <= cfg["chunk_size"]:
        raise ValueError("Execution chunk exceeds prediction chunk")
    if cfg["rollout_horizon"] < 1 or cfg["settle_steps"] < 0:
        raise ValueError("Invalid rollout/settling horizon")
    folder = output / f"shard_{shard}"
    folder.mkdir(parents=True, exist_ok=True)
    weight = run / f"{checkpoint}.pt"
    protocol = {
        **{
            key: cfg[key]
            for key in (
                "chunk_size",
                "max_action_dim",
                "action_horizon",
                "num_inference_steps",
                "rollout_horizon",
                "settle_steps",
                "eval_seed",
            )
        },
        "settle_action": [0.0] * 6 + [-1.0],
        "control_freq": 20,
        "camera_resolution": 128,
        "image_convention": "opencv",
        "action_clip": [-1, 1],
        "state": "eef_pos + quat_xyzw_to_axisangle + gripper_qpos (8D)",
        "noise_rng": "independent CUDA generator per task/init, seed=eval_seed+1000*task_id+init_id",
        "checkpoint_selection": "final_training_step"
        if checkpoint == "final"
        else "held_out_demonstration_loss",
        "video_episodes_per_task": cfg.get("video_episodes_per_task", 1),
    }
    metadata = {
        "schema": 1,
        "policy": "pi0_lora",
        "training_run": str(run),
        "checkpoint": str(weight),
        "checkpoint_sha256": _digest(weight),
        "config_sha256": _digest(run / "config.json"),
        "manifest_sha256": _digest(run / "manifest.json"),
        "evaluator_sha256": _digest(__file__),
        "episodes_per_task": episodes,
        "task_ids": [task["id"] for task in tasks],
        "shard": shard,
        "shard_task_ids": [task["id"] for task in tasks if task["id"] % 2 == shard],
        "smoke": episodes != 50 or len(tasks) != 10,
        "protocol": protocol,
    }
    metadata_path = folder / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Evaluation identity changed; choose a new output directory")
    _save_json(metadata_path, metadata)
    raw = folder / "episodes.jsonl"
    records = read_records(raw, repair_tail=True)
    done = {(row["task_id"], row["initial_state_id"]) for row in records}
    expected = {(task, initial) for task in metadata["shard_task_ids"] for initial in range(episodes)}
    if len(done) != len(records) or not done <= expected:
        raise ValueError("Duplicate or unexpected saved episodes")
    if any(row.get("checkpoint_sha256") != metadata["checkpoint_sha256"] for row in records):
        raise ValueError("Mixed checkpoint identity in resumed episodes")
    env, active_task, active_initial = None, None, None
    start = time.perf_counter()
    try:
        pending_tasks = [task for task in tasks if task["id"] % 2 == shard]
        for task in pending_tasks:
            if _digest(task["bddl"]) != task["bddl_sha256"] or _digest(task["init"]) != task["init_sha256"]:
                raise ValueError("Benchmark BDDL/initial-state files changed")
        if expected != done:
            torch.set_num_threads(cfg.get("cpu_threads", 4))
            seed_all(cfg["eval_seed"])
            saved = torch.load(weight, map_location="cpu", weights_only=False)
            if saved.get("config") != cfg:
                raise ValueError("Checkpoint config differs from the training run contract")
            if saved.get("manifest") != manifest:
                raise ValueError("Checkpoint data/normalization differs from the training manifest")
            model = Pi0Policy(cfg, manifest)
            model.load_trainable_state_dict(saved["adapter"])
            model = model.cuda().eval()
            del saved
            Env = setup_libero(cfg, folder)
            for task in pending_tasks:
                initial_states = torch.load(task["init"], map_location="cpu", weights_only=False)
                if len(initial_states) < episodes:
                    raise ValueError("Insufficient unique initial states; wrapping is forbidden")
                remaining = [initial for initial in range(episodes) if (task["id"], initial) not in done]
                if not remaining:
                    continue
                active_task = task["id"]
                env = Env(
                    bddl_file_name=task["bddl"],
                    camera_heights=128,
                    camera_widths=128,
                    controller="OSC_POSE",
                    control_freq=20,
                    hard_reset=True,
                    ignore_done=True,
                    horizon=cfg["rollout_horizon"] + cfg["settle_steps"],
                    render_gpu_device_id=int(os.environ["MUJOCO_EGL_DEVICE_ID"]),
                )
                for key in ("input_max", "input_min", "output_max", "output_min"):
                    if not np.allclose(getattr(env.env.robots[0].controller, key), task["controller"][key]):
                        raise RuntimeError(f"OSC controller {key} mismatch")
                for initial in remaining:
                    active_initial = initial
                    video = folder / f"task{task['id']:02d}_init{initial:03d}.mp4"
                    row = run_episode(
                        model,
                        env,
                        task,
                        initial_states,
                        initial,
                        cfg,
                        video_path=video if initial < cfg.get("video_episodes_per_task", 1) else None,
                        latency_path=folder / "latency.json",
                    )
                    row["checkpoint_sha256"] = metadata["checkpoint_sha256"]
                    append_record(raw, row)
                    records.append(row)
                    print(
                        json.dumps({k: row[k] for k in ("task_id", "initial_state_id", "success", "steps")}),
                        flush=True,
                    )
                env.close()
                env = None
        summary = summarize_records(
            records,
            metadata["shard_task_ids"],
            episodes,
            metadata["checkpoint_sha256"],
            cfg["rollout_horizon"],
        )
        _save_json(folder / "summary.json", summary)
        _write_csv(folder / "episodes.csv", records)
        _write_csv(folder / "task_metrics.csv", summary["per_task"])
        append_record(
            folder / "timing.jsonl",
            {"wall_seconds": time.perf_counter() - start, "resumed_episodes": len(done)},
        )
        _save_json(
            folder / "complete.json",
            {"episodes": len(records), "checkpoint_sha256": metadata["checkpoint_sha256"]},
        )
        (folder / "error.json").unlink(missing_ok=True)
        return summary
    except BaseException:
        _save_json(
            folder / "error.json",
            {
                "task_id": active_task,
                "initial_state_id": active_initial,
                "traceback": traceback.format_exc(),
                "completed_episodes": len(records),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    evaluate_shard(args.run, args.output, args.checkpoint, args.shard, args.episodes, args.max_tasks)


if __name__ == "__main__":
    main()
