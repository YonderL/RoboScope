"""Check real HF SFT preprocessing/actions and native simulator compatibility, without training."""

import argparse
import json
import sys
from pathlib import Path


def check(run):
    import numpy as np
    import torch

    from roboscope.data.smolvla_hf import HFSpatialFrames
    from roboscope.envs.pool import EnvPool
    from roboscope.policies.smolvla_rlt import build_policy, load_sft
    from roboscope.rl.collector import observation_batch
    from roboscope.runtime.common import require_gpu, save_json, seed_all

    cfg = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    require_gpu(cfg["gpu_model"])
    torch.set_num_threads(4)
    seed_all(cfg["seed"])
    base = load_sft(cfg, manifest, "cuda")
    policy = build_policy(base, cfg, manifest).cuda().eval()
    dataset = HFSpatialFrames(manifest)
    rows = []

    def compare(batch, label, native_sample=None):
        noise = torch.randn(len(batch["state"]), 50, 32, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prepared = base.prepare_batch(batch)
            if native_sample is not None:
                original = base.preprocessor(
                    {
                        **{
                            key: native_sample[key].unsqueeze(0)
                            for key in (
                                "observation.state",
                                "observation.images.image",
                                "observation.images.image2",
                            )
                        },
                        "task": [native_sample["task"]],
                    }
                )
                for key in (
                    "observation.state",
                    "observation.images.image",
                    "observation.images.image2",
                    "observation.language.tokens",
                    "observation.language.attention_mask",
                ):
                    torch.testing.assert_close(prepared[key], original[key], rtol=0, atol=0)
                dataset_actions = base.postprocessor(base.policy.predict_action_chunk(original, noise=noise))
            native = base.postprocessor(base.policy.predict_action_chunk(prepared, noise=noise))
            actual = base.predict(batch, noise=noise)
            state, reference = policy.describe(batch, noise=noise)
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        if native_sample is not None:
            torch.testing.assert_close(actual, dataset_actions, rtol=1e-4, atol=1e-5)
        expected = native[:, : cfg["action_horizon"]].cuda().float().clamp(-1, 1)
        torch.testing.assert_close(reference, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            state[:, cfg["token_dim"] : -1], prepared["observation.state"], rtol=1e-5, atol=1e-6
        )
        if not torch.isfinite(state).all() or not torch.isfinite(reference).all():
            raise RuntimeError("Nonfinite HF policy output")
        rows.append(
            {
                "sample": label,
                "reference_max_abs_error": (reference - expected).abs().max().item(),
                "state_shape": list(state.shape),
                "reference_shape": list(reference.shape),
            }
        )
        return actual[0, 0].float().cpu().numpy().clip(-1, 1)

    for index in (0, len(dataset) // 2, len(dataset) - 1):
        batch = {key: value.unsqueeze(0).cuda() for key, value in dataset[index].items()}
        compare(batch, f"dataset_frame_{index}", dataset.dataset[index])
    pool = EnvPool(cfg, run, 1)
    try:
        for task_id in (0, 5):
            task = manifest["tasks"][task_id]
            pool.connections[0].send(("reset", (task, 0, "")))
            obs, _ = pool.receive(0)
            action = compare(observation_batch([obs], task, [0], cfg, "cuda"), f"fixed_task_{task_id}")
            pool.connections[0].send(("step", action))
            obs, _ = pool.receive(0)
            assert obs["state"].shape == (8,) and np.isfinite(obs["state"]).all()
        task = manifest["tasks"][0]
        pool.connections[0].send(("reset_training", (task, cfg["rollout_seed"], "")))
        obs, _ = pool.receive(0)
        compare(observation_batch([obs], task, [0], cfg, "cuda"), "random_training_reset")
    finally:
        pool.close()
    save_json(
        run / "alignment.json",
        {
            "passed": True,
            "checks": rows,
            "sft_checkpoint": cfg["sft_checkpoint"],
            "sft_sha256": cfg["sft_sha256"],
            "environment_versions": cfg["environment_versions"],
            "gpu": torch.cuda.get_device_name(0),
            "scope": "real checkpoint, dataset preprocessing, native actions, fixed/random simulator resets",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        check(args.output)
        return
    from roboscope.data.smolvla_hf import prepare_contract
    from roboscope.runtime.common import save_json
    from roboscope.runtime.gpu import cards_and_envs, run_workers
    from roboscope.workflows.config import validate_rlt

    cfg = json.loads(args.recipe.read_text())
    validate_rlt(cfg)
    cfg, manifest = prepare_contract(args.source, cfg)
    run = args.output.resolve()
    cards, envs = cards_and_envs(cfg["gpu_model"])
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "config.json", cfg)
    save_json(run / "manifest.json", manifest)
    save_json(run / "gpu_mapping.json", cards)
    envs[0].update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_CACHE=str(run / "hf_cache"),
        NUMBA_CACHE_DIR=str(run / "numba_cache"),
        TOKENIZERS_PARALLELISM="false",
    )
    command = [
        sys.executable,
        "-m",
        "roboscope.workflows.smolvla_rlt_check",
        "--worker",
        "--output",
        str(run),
    ]
    run_workers([command], envs[:1], [run / "alignment.log"])
    print((run / "alignment.json").read_text())


if __name__ == "__main__":
    main()
