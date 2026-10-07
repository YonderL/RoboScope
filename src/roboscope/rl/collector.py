"""LIBERO training rollouts, independently seeded and separate from fixed-state evaluation."""

import numpy as np
import torch

from roboscope.rl.replay import episode_transitions
from roboscope.rl.rewards import progress_enabled
from roboscope.runtime.common import CAMERAS


def observation_batch(observations, task, steps, cfg, device):
    batch = {
        "state": torch.from_numpy(np.stack([obs["state"] for obs in observations])).to(device),
        "task_id": torch.full((len(observations),), task["id"], device=device, dtype=torch.long),
        "remaining_steps": torch.tensor([cfg["rollout_horizon"] - s for s in steps], device=device),
    }
    for camera in CAMERAS:
        batch[camera] = (
            torch.from_numpy(np.stack([obs[camera] for obs in observations])).permute(0, 3, 1, 2).to(device)
        )
    return batch


def collect_episode(pool, policy, task, episode_id, cfg, device, warmup):
    """Complete one episode before committing replay or changing learner weights.

    Seeds and missing intermediate VLA references depend on episode/step, not
    feature extraction batch size. This also makes episode-boundary resume safe.
    """
    seed = cfg["rollout_seed"] + episode_id
    # 探索噪声与 VLA flow 噪声分开管理；固定 episode/step 可复做中断的轨迹。
    generator = torch.Generator(device=device).manual_seed(seed)
    reference_generator = torch.Generator(device=device)
    pool.connections[0].send(("reset_training", (task, seed, "")))
    # reset_training 使用随机训练初态，正式评测的固定初态不进入 replay。
    obs, _ = pool.receive(0)
    observations, actions, features = [obs], [], {}

    def describe(indices):
        batch = observation_batch([observations[i] for i in indices], task, indices, cfg, device)
        noise = torch.stack(
            [
                torch.randn(50, 32, device=device, generator=reference_generator.manual_seed(seed * 1009 + i))
                for i in indices
            ]
        )
        with (
            torch.inference_mode(),
            torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["amp"] and device.type == "cuda"),
        ):
            state, reference = policy.describe(batch, noise)
        for i, state_row, ref_row in zip(
            indices, state.float().cpu().numpy(), reference.float().cpu().numpy(), strict=True
        ):
            features[i] = (state_row, ref_row)
        return state, reference

    success = False
    while len(actions) < cfg["rollout_horizon"] and not success:
        state, reference = describe([len(actions)])
        with torch.inference_mode():
            # warmup 直接执行冻结 SFT；之后由 actor 直接输出 C 步动作。
            # actor 并非 reference + residual，因此接管时不自动继承 SFT 成功率。
            chunk = reference if warmup else policy.actor.sample(state, reference, generator=generator)
        chunk = chunk[0].float().cpu().numpy()
        if not np.isfinite(chunk).all():
            raise RuntimeError("Nonfinite RLT action")
        for action in chunk:
            action = np.clip(action, -1, 1)
            pool.connections[0].send(("step", action))
            obs, success = pool.receive(0)
            actions.append(action.copy())
            observations.append(obs)
            if success or len(actions) == cfg["rollout_horizon"]:
                break
    # Paper's stride-2 windows include chunks crossing policy call boundaries,
    # but NEVER episode boundaries. Generate aligned reference actions for
    # those intermediate states, rather than slicing a stale earlier proposal.
    required = set(range(0, len(actions), cfg["replay_stride"]))
    required |= {i + cfg["action_horizon"] for i in required if i + cfg["action_horizon"] < len(actions)}
    missing = sorted(required - features.keys())
    # 例如 t=2 不在 10 步推理边界上，需要用当时 RGB 重算 reference。
    # 此过程包含完整 flow sampling，是 stride-2 replay 的主要额外计算开销。
    for start in range(0, len(missing), cfg["feature_batch_size"]):
        describe(missing[start : start + cfg["feature_batch_size"]])
    potentials, reward_metrics = None, {}
    if progress_enabled(cfg):
        # Fail closed if any physical step missed the privileged reward channel.
        progress = [obs["reward_progress"] for obs in observations]
        potentials = [row["potential"] for row in progress]
        pickup = np.asarray([row["pickup"] > 0 for row in progress])
        phi = np.asarray(potentials, dtype=np.float64)
        phi[-1] = 0.0
        per_step_shaping = cfg["reward"]["scale"] * (cfg["gamma"] * phi[1:] - phi[:-1])
        reward_metrics = {
            "reward_mode": cfg["reward"]["mode"],
            "potential_max": max(potentials),
            "pickup_proxy_steps": int(pickup[1:].sum()),
            "pickup_proxy_entries": int((~pickup[:-1] & pickup[1:]).sum()),
            "pickup_proxy_exits": int((pickup[:-1] & ~pickup[1:]).sum()),
            "sparse_return": float(success),
            "shaping_return": float(per_step_shaping.sum()),
            "discounted_shaping_return": float(
                np.dot(cfg["gamma"] ** np.arange(len(actions)), per_step_shaping)
            ),
        }
    rows = list(
        episode_transitions(
            actions, success, features, cfg, task["id"], episode_id, seed, warmup, potentials=potentials
        )
    )
    return rows, {
        "episode_id": episode_id,
        "task_id": task["id"],
        "seed": seed,
        "steps": len(actions),
        "success": int(success),
        "warmup": warmup,
        "replay_transitions": len(rows),
        "reset": "randomized_training_reset_no_fixed_eval_initial_state",
        **reward_metrics,
    }
