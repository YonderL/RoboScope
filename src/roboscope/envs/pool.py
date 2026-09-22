"""Spawned simulator workers; policies remain in the parent process."""

import multiprocessing as mp
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def compact_observation(obs):
    # 只传输策略真正需要的数组，避免 IPC 复制大量仿真内部状态。
    return {
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy(),
        "state": np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32),
        "agentview_rgb": obs["agentview_image"].copy(),
        "eye_in_hand_rgb": obs["robot0_eye_in_hand_image"].copy(),
    }


def env_worker(connection, cfg, directory, slot):
    """子进程仅负责仿真/渲染，不加载 ACT，也不在 GPU 上保存模型副本。"""
    env, writer, task_name = None, None, None
    try:
        torch.set_num_threads(1)
        from roboscope.envs.libero import setup_libero, video_frame

        folder = Path(directory) / f"env_worker_{slot}"
        folder.mkdir(parents=True, exist_ok=True)
        Env = setup_libero(cfg, folder)
        initial_states = {}
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command in ("reset", "reset_training"):
                task, initial_id, video_path = payload
                training = command == "reset_training"
                episode_seed = initial_id if training else cfg["eval_seed"] + task["id"] * 1000 + initial_id
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                if task_name != task["name"]:
                    if env is not None:
                        env.close()
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
                    task_name = task["name"]
                    for key in ("input_max", "input_min", "output_max", "output_min"):
                        if not np.allclose(
                            getattr(env.env.robots[0].controller, key), task["controller"][key]
                        ):
                            raise RuntimeError(f"Controller {key} mismatch")
                if not training and task_name not in initial_states:
                    initial_states[task_name] = torch.load(
                        task["init"], map_location="cpu", weights_only=False
                    )
                # 每次 hard reset 都重新设置 seed，结果不依赖哪个进程领取 episode。
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                env.seed(episode_seed)
                obs = env.reset()
                if not training:
                    obs = env.set_init_state(np.asarray(initial_states[task_name][initial_id]))
                for _ in range(cfg["settle_steps"]):
                    obs, _, _, _ = env.step(np.zeros(7))
                steps = 0
                if video_path:
                    import imageio.v2 as imageio

                    writer = imageio.get_writer(
                        video_path,
                        fps=20,
                        codec="libx264",
                        macro_block_size=1,
                        ffmpeg_params=["-threads", "1"],
                    )
                    writer.append_data(video_frame(obs))
                connection.send(("ok", compact_observation(obs), False))
            elif command == "step":
                obs, _, _, _ = env.step(payload)
                steps += 1
                success = bool(env.check_success())
                if writer is not None:
                    writer.append_data(video_frame(obs))
                    if success or steps >= cfg["rollout_horizon"]:
                        writer.close()
                        writer = None
                connection.send(("ok", compact_observation(obs), success))
            else:
                raise ValueError(command)
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


class EnvPool:
    def __init__(self, cfg, directory, size):
        context = mp.get_context("spawn")  # CUDA/EGL 已初始化时不能 fork。
        self.connections, self.processes = [], []
        try:
            for slot in range(size):
                parent, child = context.Pipe()
                proc = context.Process(
                    target=env_worker, args=(child, cfg, str(directory), slot), daemon=True
                )
                proc.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(proc)
        except BaseException:
            self.close()
            raise

    def receive(self, slot):
        conn = self.connections[slot]
        if not conn.poll(180):
            raise TimeoutError(f"Environment {slot} did not respond within 180 seconds")
        status, observation, success = conn.recv()
        if status == "error":
            raise RuntimeError(f"Environment {slot} failed:\n{observation}")
        return observation, success

    def close(self):
        for conn in self.connections:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 8
        for proc in self.processes:
            proc.join(timeout=max(0, deadline - time.monotonic()))
            if proc.is_alive():
                proc.terminate()
        for proc in self.processes:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        for conn in self.connections:
            conn.close()
