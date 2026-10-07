"""Native LeRobot/hf-libero environment contract for new HF Spatial runs."""

import hashlib
import random
import traceback
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import torch

from roboscope.rl.rewards import progress_enabled

FIXED_RESET_PROTOCOL = "property_sampler_clear_v1"


def reset_fixed_state(env, seed, protocol):
    """One property-sampler set per hard reset, regardless of worker history."""
    if protocol != FIXED_RESET_PROTOCOL:
        raise ValueError(f"Unsupported HF fixed-state reset protocol: {protocol}")
    if not env.hard_reset or not env.init_states:
        raise ValueError("Fixed-state reset protocol requires hard_reset and benchmark init_states")
    env._ensure_env()
    # hf-libero 0.1.4 appends Open/Close samplers each time _load_model runs.
    # Each duplicate consumes RNG before fixture placement, whose model poses
    # are not restored by the fixed qpos/qvel state. Rebuild exactly one set.
    env._env.env.object_property_initializers.clear()
    return env.reset(seed=seed)


def reset_state_digest(env):
    """Hash settled physics state AND model fixture poses for paired evaluation."""
    sim = env._env.env.sim
    digest = hashlib.sha256()
    for array in (sim.get_state().flatten(), sim.model.body_pos, sim.model.body_quat):
        values = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def check_runtime(cfg):
    expected = cfg["environment_versions"]
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"HF LIBERO runtime mismatch: expected {expected}, found {actual}")
    # Reject a PYTHONPATH checkout shadowing the installed hf-libero package.
    import importlib.util

    origin = Path(importlib.util.find_spec("libero").origin).resolve()
    installed = distribution("hf-libero")
    if origin not in {Path(installed.locate_file(p)).resolve() for p in installed.files or []}:
        raise ValueError(f"libero is not supplied by installed hf-libero: {origin}")
    return actual


def setup(cfg, directory):
    check_runtime(cfg)
    # Set the asset paths before importing LIBERO (which otherwise prompts).
    from roboscope.envs.libero import setup_libero

    setup_libero(cfg, directory)
    import robosuite.macros as macros
    from libero.libero.benchmark import get_benchmark_dict

    macros.IMAGE_CONVENTION = "opengl"
    return get_benchmark_dict()["libero_spatial"]()


def compact_observation(obs):
    """Use the native processor's exact quaternion conversion and image rotation."""
    from lerobot.processor.env_processor import LiberoProcessorStep

    robot = obs["robot_state"]
    prepared = {
        "observation.robot_state": {
            group: {key: torch.as_tensor(value).unsqueeze(0) for key, value in fields.items()}
            for group, fields in {
                "eef": {"pos": robot["eef"]["pos"], "quat": robot["eef"]["quat"]},
                "gripper": {"qpos": robot["gripper"]["qpos"]},
            }.items()
        }
    }
    for key in ("image", "image2"):
        pixels = np.asarray(obs["pixels"][key])
        if pixels.shape != (256, 256, 3) or pixels.dtype != np.uint8:
            raise ValueError("HF LIBERO requires 256x256 uint8 RGB")
        prepared[f"observation.images.{key}"] = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None]
    processed = LiberoProcessorStep()._process_observation(prepared)
    return {
        "state": processed["observation.state"][0].numpy().copy(),
        "eef_pos": np.asarray(robot["eef"]["pos"], dtype=np.float32).copy(),
        **{
            target: processed[f"observation.images.{source}"][0].permute(1, 2, 0).numpy().copy()
            for source, target in (("image", "agentview_rgb"), ("image2", "eye_in_hand_rgb"))
        },
    }


def env_worker(connection, cfg, directory, slot):
    """Native resets/settling/controller, with separate random-training and fixed-eval modes."""
    env = writer = None
    current = None
    progress = None
    try:
        shaped_reward = progress_enabled(cfg)
        torch.set_num_threads(1)
        folder = Path(directory) / f"env_worker_{slot}"
        folder.mkdir(parents=True, exist_ok=True)
        suite = setup(cfg, folder)
        from lerobot.envs.libero import LiberoEnv

        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command in ("reset", "reset_training"):
                task, initial, video = payload
                training = command == "reset_training"
                seed = initial if training else cfg["eval_seed"] + task["id"] * 1000 + initial
                random.seed(seed)
                np.random.seed(seed)
                if writer is not None:
                    writer.close()
                    writer = None
                if current != (task["id"], training):
                    if env is not None:
                        env.close()
                    if suite.get_task(task["id"]).name != task["name"]:
                        raise ValueError("Native task order changed")
                    env = LiberoEnv(
                        suite,
                        task["id"],
                        "libero_spatial",
                        obs_type="pixels_agent_pos",
                        observation_width=256,
                        observation_height=256,
                        init_states=not training,
                        episode_length=cfg["rollout_horizon"],
                        num_steps_wait=cfg["settle_steps"],
                        control_freq=20,
                        control_mode="relative",
                        hard_reset=True,
                    )
                    current = (task["id"], training)
                if not training:
                    if not 0 <= initial < len(env._init_states):
                        raise ValueError("Fixed initial-state wrapping is forbidden")
                    env.init_state_id = initial
                protocol = cfg.get("hf_eval_reset_protocol") if not training else None
                obs, _ = reset_fixed_state(env, seed, protocol) if protocol else env.reset(seed=seed)
                compact = compact_observation(obs)
                if protocol:
                    compact["reset_state_sha256"] = reset_state_digest(env)
                progress = None
                if training and shaped_reward:
                    from roboscope.envs.libero_reward import LiberoGraspProgress

                    progress = LiberoGraspProgress(env._env.env, cfg["reward"])
                steps, success = 0, False
                if video:
                    import imageio.v2 as imageio

                    writer = imageio.get_writer(
                        video, fps=20, codec="libx264", macro_block_size=1, ffmpeg_params=["-threads", "1"]
                    )
            elif command == "step":
                obs, _, terminated, truncated, info = env.step(payload)
                steps += 1
                success = bool(info["is_success"])
                if (terminated or truncated) and not success and steps < cfg["rollout_horizon"]:
                    raise RuntimeError("Unexpected early native environment termination")
                compact = compact_observation(obs)
            else:
                raise ValueError(command)
            if progress is not None:
                # Kept out of observation_batch: reward-only simulator truth.
                compact["reward_progress"] = progress.measure(compact["eef_pos"])
            if writer is not None:
                writer.append_data(np.concatenate([compact["agentview_rgb"], compact["eye_in_hand_rgb"]], 1))
                if success or steps >= cfg["rollout_horizon"]:
                    writer.close()
                    writer = None
            connection.send(("ok", compact, success))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc(), False))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        connection.close()
