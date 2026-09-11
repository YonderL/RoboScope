"""Public entrypoint. Expensive operations are preview-only unless --start is supplied."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="roboscope")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train a recipe (preview by default)")
    training.add_argument("--recipe", type=Path, required=True)
    training.add_argument("--data-root", type=Path, required=True, help="Parent containing libero_spatial/")
    training.add_argument(
        "--libero-root", type=Path, required=True, help="Folder containing assets/, bddl_files/, init_files/"
    )
    training.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate", help="One checkpoint and execution configuration")
    evaluation.add_argument("--source", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--checkpoint", choices=["best", "final"], default="final")
    evaluation.add_argument("--episodes", type=int, default=50)
    evaluation.add_argument("--ddim-steps", type=int, choices=[5, 10, 20, 50, 100], default=10)
    evaluation.add_argument("--ta", type=int, choices=[1, 4, 8], default=8)
    for p in (training, evaluation):
        p.add_argument("--start", action="store_true")
        p.add_argument("--resume", action="store_true")
    report = sub.add_parser("report", help="Render figures from portable audited records; no GPU")
    report.add_argument("--data", type=Path, default=Path("results/libero_spatial"))
    report.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    if args.command == "report":
        from roboscope.reporting.figures import render

        render(args.data, args.output)
    elif args.command == "train":
        cfg = json.loads(args.recipe.read_text())
        from roboscope.workflows.config import validate_recipe

        validate_recipe(cfg)
        cfg.update(
            data_root=str(args.data_root.resolve()),
            libero_root=str(args.libero_root.resolve()),
            output_root=str(args.output.resolve()),
        )
        print(json.dumps(cfg, indent=2))
        if args.start:
            from roboscope.workflows.experiments import train

            train(cfg, args.resume)
        else:
            print("Preview only; pass --start to execute on the two RTX 4090s.")
    else:
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        if args.start:
            from roboscope.workflows.experiments import evaluate

            evaluate(
                args.source,
                args.output,
                args.checkpoint,
                args.episodes,
                args.ddim_steps,
                args.ta,
                args.resume,
            )
        else:
            print("Preview only; pass --start to evaluate. No checkpoint loaded.")
