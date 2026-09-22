"""Two-stage RLT: frozen-SFT token reconstruction, then rollout-only online RL."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from roboscope.data.libero import FrameDataset
from roboscope.data.sequences import StepBatchSampler
from roboscope.policies.smolvla_rlt import build_policy, load_sft, vlm_features
from roboscope.rl.collector import collect_episode
from roboscope.rl.learner import RLTAgent
from roboscope.rl.replay import ReplayBuffer
from roboscope.runtime.common import atomic_save, require_4090, save_json, seed_all
from roboscope.trainers.pi0 import capture_rng, cpu_tree, manifest_digest, move_batch, restore_rng


def train_token(run, cfg, manifest, policy, device, resume=False):
    """Stage one uses only training demonstrations; alpha=0 keeps SFT frozen."""
    # 第一阶段只学观测压缩：SFT 演示可用于重建，但不会成为第二阶段 replay 数据。
    identity = manifest_digest(manifest)
    final, last = run / "token.pt", run / "token_last.pt"
    if final.exists():
        if not resume:
            raise FileExistsError("Token already trained; use --resume")
        saved = torch.load(final, map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["manifest_sha256"] != identity:
            raise ValueError("RL token identity changed")
        policy.token.load_state_dict(saved["token"], strict=True)
        return
    optimizer = torch.optim.AdamW(policy.token.parameters(), lr=cfg["token_lr"])
    step = 0
    if resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["manifest_sha256"] != identity:
            raise ValueError("RL token resume identity changed")
        policy.token.load_state_dict(saved["token"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], device)
        step = saved["step"]
        if step >= cfg["token_steps"]:
            atomic_save(
                final, {key: value for key, value in saved.items() if key not in ("optimizer", "rng")}
            )
            return
    dataset = FrameDataset(manifest, 1, "train", run / "image_cache")
    kwargs = dict(num_workers=cfg["workers"], pin_memory=device.type == "cuda")
    if cfg["workers"]:
        kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(
        dataset,
        batch_sampler=StepBatchSampler(
            len(dataset), cfg["token_batch_size"], cfg["token_steps"], cfg["seed"], step
        ),
        generator=torch.Generator().manual_seed(cfg["seed"] + 101),
        **kwargs,
    )
    policy.base.eval()
    policy.token.train()
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for start in range(0, len(batch["state"]), cfg["feature_batch_size"]):
            # 分小批提取冻结 VLM 特征以控制显存；累积梯度后才执行一次 token 更新。
            part = {key: value[start : start + cfg["feature_batch_size"]] for key, value in batch.items()}
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=cfg["amp"] and device.type == "cuda"
            ):
                features, valid = vlm_features(policy.base, part)
            loss = policy.token(features, valid) * (len(part["state"]) / len(batch["state"]))
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(policy.token.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % cfg["log_every"] == 0 or step == cfg["token_steps"]:
            print(json.dumps({"stage": "token", "step": step, "reconstruction_loss": loss_sum}), flush=True)
        if step % cfg["token_save_every"] == 0 or step == cfg["token_steps"]:
            payload = {
                "token": cpu_tree(policy.token.state_dict()),
                "step": step,
                "config": cfg,
                "manifest_sha256": identity,
                "reconstruction_loss": loss_sum,
            }
            atomic_save(
                last, {**payload, "optimizer": cpu_tree(optimizer.state_dict()), "rng": capture_rng(device)}
            )
            if step == cfg["token_steps"]:
                atomic_save(final, payload)


def train_online(
    run, cfg, manifest, policy, pool, device, resume=False, collector=collect_episode, collect_only=False
):
    """Commit replay/agent/RNG together at episode boundaries.

    An interruption discards only the uncommitted episode and its updates; the
    next run repeats that episode from its seed. No half-episode enters replay.
    """
    policy.base.eval().requires_grad_(False)
    policy.token.eval().requires_grad_(False)
    # 在线阶段不能再改 token：否则 replay 里的旧向量和新向量不在同一个表示空间。
    seed_all(cfg["seed"] + 2000)
    agent = RLTAgent(cfg).to(device)
    replay = ReplayBuffer(cfg["replay_capacity"], cfg["seed"] + 17)
    env_steps, episode_id, records, warmup_done = 0, 0, [], False
    identity = manifest_digest(manifest)
    last = run / "last.pt"
    if resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["manifest_sha256"] != identity:
            raise ValueError("RLT config or manifest changed on resume")
        policy.token.load_state_dict(saved["token"], strict=True)
        agent.restore(saved["agent"])
        replay.load_state_dict(saved["replay"])
        restore_rng(saved["rng"], device)
        env_steps, episode_id, records = saved["env_steps"], saved["episode_id"], saved["records"]
        warmup_done = saved["warmup_done"]
    elif last.exists():
        raise FileExistsError("Existing RLT checkpoint; use --resume")
    if collect_only and warmup_done:
        raise ValueError("Online learning already started; cannot return to warmup collection")
    policy.actor = agent.actor

    def export():
        return {
            "format": "roboscope.smolvla_rlt.v1",
            "config": cfg,
            "manifest_sha256": identity,
            "token": cpu_tree(policy.token.state_dict()),
            "actor": cpu_tree(agent.actor.state_dict()),
            "env_steps": env_steps,
            "episode_id": episode_id,
            "updates": agent.updates,
        }

    def warmup_ready():
        return env_steps >= cfg["warmup_steps"] and len(replay) >= cfg["batch_size"]

    def learn(new_transitions):
        nonlocal warmup_done
        updates = new_transitions * cfg["updates_per_transition"]
        if not warmup_done:
            # 先在 SFT rollout 上做若干次更新，再让新 actor 接管。
            # 这不是纯 BC 预训练，也没有通过成功率门槛才接管的保护机制。
            updates += cfg["initial_updates"]
        metrics = {}
        agent.train()
        # UTD 按新增 transition 计数：600 步、stride=2 -> 300 条 -> 1500 次更新。
        # 采集与学习同步运行；更新期间环境不前进，GPU 利用率会呈阶段性变化。
        for _ in range(updates):
            metrics = agent.update(replay.sample(cfg["batch_size"], device))
        warmup_done = True
        return metrics

    def save_progress(metrics):
        records[-1].update(
            # metrics 仅是该回合最后一次 learner 更新，不是所有更新的平均值。
            env_steps=env_steps,
            replay_size=len(replay),
            replay_rewarded_transitions=int((replay.arrays["reward"][: len(replay)] > 0).sum()),
            updates=agent.updates,
            **{key: value for key, value in metrics.items() if key != "updates"},
        )
        payload = export()
        atomic_save(
            last,
            # agent、replay 与 RNG 一起提交，恢复时从上一个完整 episode 边界继续。
            {
                **payload,
                "agent": cpu_tree(agent.training_state()),
                "replay": replay.state_dict(),
                "rng": capture_rng(device),
                "records": records,
                "warmup_done": warmup_done,
            },
        )
        save_json(run / "rollouts.json", records)
        print(json.dumps({"stage": "warmup" if collect_only else "online", **records[-1]}), flush=True)

    save_json(run / "rollouts.json", records)
    # 分阶段采集只保存 replay，不更新网络。恢复在线阶段时，先补做一键流程
    # 原本会在 warmup 末尾做的更新，再允许 actor 接管；两种运行方式保持一致。
    if not collect_only and not warmup_done and warmup_ready():
        save_progress(learn(records[-1]["replay_transitions"]))
    while env_steps < cfg["online_steps"] and not (collect_only and warmup_ready()):
        task = manifest["tasks"][episode_id % len(manifest["tasks"])]
        warmup = not warmup_done
        policy.eval()
        transitions, record = collector(pool, policy, task, episode_id, cfg, device, warmup)
        for row in transitions:
            replay.add(row)
        env_steps += record["steps"]
        episode_id += 1
        record["replay_transitions"] = len(transitions)
        records.append(record)
        metrics = learn(len(transitions)) if not collect_only and warmup_ready() else {}
        save_progress(metrics)
    if collect_only:
        if not warmup_ready():
            raise RuntimeError("Budget ended before warmup had enough steps and replay samples")
        return records
    if not warmup_done:
        raise RuntimeError(
            "Budget ended without RL updates; increase online_steps or reduce warmup/batch size"
        )
    atomic_save(run / "final.pt", export())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["all", "token", "warmup", "online"], default="all")
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    if args.stage in ("warmup", "online") and not (args.run / "token.pt").exists():
        raise FileNotFoundError("Train and freeze the RL token first with --stage token")
    if args.stage == "online" and not (args.run / "last.pt").exists():
        raise FileNotFoundError("Collect the warmup replay first with --stage warmup")
    require_4090()
    device = torch.device("cuda")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    seed_all(cfg["seed"])
    base = load_sft(cfg, manifest, device)
    policy = build_policy(base, cfg, manifest).to(device)
    train_token(args.run, cfg, manifest, policy, device, args.resume)
    if args.stage == "token":
        return
    from roboscope.envs.pool import EnvPool

    pool = EnvPool(cfg, args.run, 1)
    try:
        train_online(
            args.run, cfg, manifest, policy, pool, device, args.resume, collect_only=args.stage == "warmup"
        )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
