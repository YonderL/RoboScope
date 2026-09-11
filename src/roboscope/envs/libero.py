"""LIBERO observation and rendering setup; no policy or trainer dependency."""

import os
from pathlib import Path

import numpy as np


def setup_libero(cfg, run):
    # Keep the pip hf-libero implementation, and use the user's existing benchmark assets.
    # Never modify ~/.libero or the original LIBERO checkout.
    import yaml

    root = Path(cfg["libero_root"])
    folder = run / "libero_config"
    folder.mkdir(exist_ok=True)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(root),
                "bddl_files": str(root / "bddl_files"),
                "init_states": str(root / "init_files"),
                "assets": str(root / "assets"),
                "datasets": cfg["data_root"],
            }
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(folder)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import robosuite.macros as macros

    macros.IMAGE_CONVENTION = "opencv"
    import libero.libero as libero_api

    # hf-libero 0.1.4 的 get_assets_path() 不读取 config.yaml 的 assets 字段，
    # 而是先查包内资源再下载。明确设置进程内缓存，复用本地完整资源。
    # 不写 site-packages，也不触发额外数据下载。
    if hasattr(libero_api, "_assets_path_cache"):
        libero_api._assets_path_cache = str(root / "assets")
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def video_frame(obs):
    return np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand_image"]], axis=1).copy()
