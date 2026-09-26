"""Pi-0 preparation, two-GPU DDP training and separate two-shard evaluation."""

import json
import os
import sys
from pathlib import Path

from roboscope.workflows.experiments import lock_run, save_contract


def train_pi0(cfg, resume=False):
    from roboscope.data.cache import prepare_cache
    from roboscope.data.libero import save_json
    from roboscope.data.pi0 import prepare_pi0
    from roboscope.policies.pi0 import resolve_assets
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    run = Path(cfg["output_root"])
    lock = lock_run(run)
    try:
        cards, _ = cards_and_envs()
        manifest = prepare_pi0(cfg)
        cfg = dict(cfg)
        asset_keys = ("pretrained_path", "pretrained_revision", "tokenizer_path", "tokenizer_revision")
        requested = {key: cfg[key] for key in asset_keys}
        if (run / "config.json").exists():
            if not resume:
                raise ValueError("Existing Pi-0 run; pass --resume or choose another output")
            saved = json.loads((run / "config.json").read_text())
            if saved["asset_requests"] != requested:
                raise ValueError("Pretrained asset request changed; use a new output")
            cfg.update({key: saved[key] for key in asset_keys})
        else:
            # Download once before ranks are launched; immutable revisions and
            # local paths travel with the run. No random-weight fallback.
            cfg.update(resolve_assets(cfg))
        cfg["asset_requests"] = requested
        save_contract(run, cfg, manifest, resume)
        save_json(run / "gpu_mapping.json", cards)
        prepare_cache(manifest, run / "image_cache")
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=",".join(card["uuid"] for card in cards),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS=str(cfg["cpu_threads"]),
            MKL_NUM_THREADS=str(cfg["cpu_threads"]),
            OPENBLAS_NUM_THREADS="1",
            # The two consumer Ada GPUs have no NVLink. Avoid unsupported
            # peer-to-peer paths; NCCL still performs synchronous DDP via SHM.
            NCCL_P2P_DISABLE="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "-m",
            "roboscope.trainers.pi0",
            "--run",
            str(run),
        ]
        if resume:
            command.append("--resume")
        print(
            f"Pi-0 DDP on physical GPUs {[c['index'] for c in cards]}; log: {run / 'train.log'}", flush=True
        )
        run_workers([command], [environment], [run / "train.log"])
    finally:
        lock.close()


def evaluate_pi0(
    source,
    output,
    checkpoint="final",
    episodes=50,
    resume=False,
    libero_root=None,
    gpu_model=None,
    task_ids=None,
):
    import torch

    from roboscope.data.libero import digest, save_json
    from roboscope.evaluation.pi0 import evaluation_context
    from roboscope.runtime.gpu import cards_and_envs, run_workers

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Pi-0 evaluation requires a separate output directory")
    cfg = json.loads((source / "config.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    _, tasks = evaluation_context(cfg, manifest, libero_root)
    weight = source / f"{checkpoint}.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    for task in tasks:
        if len(torch.load(task["init"], weights_only=False, map_location="cpu")) < episodes:
            raise ValueError("Insufficient unique initial states; wrapping is forbidden")
    contract = {
        "source": str(source),
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest(weight),
        "episodes_per_task": episodes,
        "source_config": cfg,
        "libero_root": str(Path(libero_root).resolve()) if libero_root else None,
        "gpu_model": gpu_model or cfg.get("gpu_model", "4090"),
        "task_ids": task_ids,
    }
    lock = lock_run(output)
    try:
        path = output / "evaluation_config.json"
        if path.exists():
            if not resume or json.loads(path.read_text()) != contract:
                raise ValueError("Existing or changed Pi-0 evaluation; use --resume with the same contract")
        else:
            if set(p.name for p in output.iterdir()) != {".launcher.lock"}:
                raise ValueError("Evaluation output is not empty")
            save_json(path, contract)
        cards, envs = cards_and_envs(gpu_model or cfg.get("gpu_model", "4090"))
        save_json(output / "gpu_mapping.json", cards)
        commands = [
            [
                sys.executable,
                "-m",
                "roboscope.evaluation.pi0",
                "--run",
                str(source),
                "--output",
                str(output),
                "--checkpoint",
                checkpoint,
                "--episodes",
                str(episodes),
                "--shard",
                str(shard),
            ]
            for shard in range(2)
        ]
        for command in commands:
            if libero_root:
                command += ["--libero-root", str(Path(libero_root).resolve())]
            if gpu_model:
                command += ["--gpu-model", gpu_model]
            if task_ids is not None:
                command += ["--task-ids", *map(str, task_ids)]
        run_workers(commands, envs, [output / f"eval_shard{i}.log" for i in range(2)])
        from roboscope.evaluation.pi0 import aggregate_evaluation

        aggregate_evaluation(output)
    finally:
        lock.close()
