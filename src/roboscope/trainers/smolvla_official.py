"""Configure local LIBERO assets and BF16, then use the unmodified LeRobot trainer."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader as _DataLoader

    from roboscope.runtime.common import require_4090

    class _DropLastLoader(_DataLoader):
        """Keep every compiled batch at 64. The epoch tail is only 42 frames."""

        def __init__(self, *args, **kwargs):
            if kwargs.get("shuffle") and kwargs.get("batch_size", 1) > 1:
                kwargs["drop_last"] = True
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = _DropLastLoader
    require_4090()
    torch.set_num_threads(4)
    # SmolVLA has use_amp but no dtype field. Native LeRobot reads this autocast
    # default when constructing Accelerate; explicitly request BF16, not FP16.
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    import libero.libero as libero_api
    import robosuite.macros as macros

    libero_api._assets_path_cache = str(args.libero_root / "assets")
    # The native LiberoProcessorStep rotates OpenGL images by 180 degrees to
    # match HuggingFaceVLA/libero. Do not apply the old adapter's vertical flip.
    macros.IMAGE_CONVENTION = "opengl"
    if args.evaluate:
        if args.checkpoint is None or not (args.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError("Evaluation requires an existing native LeRobot checkpoint")
        sys.argv = [
            "lerobot-eval",
            f"--config_path={args.config}",
            f"--policy.path={args.checkpoint}",
        ]
        runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")
    else:
        sys.argv = ["lerobot-train", f"--config_path={args.config}"]
        if args.resume:
            sys.argv.append("--resume=true")
        runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")


if __name__ == "__main__":
    main()
