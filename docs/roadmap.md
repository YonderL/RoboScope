# Roadmap with acceptance criteria

| Stage | Status | Acceptance criterion |
|---|---|---|
| ACT / DP on LIBERO-Spatial | Implemented; measured in archived runs | Audited per-episode results, matched comparison, documented limitations |
| Package migration | Implemented; bounded validation | No dependency on local experiment scripts; model/data parity checks; source release builds |
| Stronger evaluation protocol | Planned | Fixed batch replay check, failure annotations, development/test initial-state separation, additional training seeds |
| Pi-0 LoRA supervised fine-tuning | Pipeline implemented; pretrained full-run validation pending asset download | [Two-4090 runbook](pi0_lora_spatial.md); real language inputs, separate EEF state adapter, resumable DDP and fixed-init evaluation; no measured baseline claimed |
| SmolVLA RLT post-training | Implemented; full Spatial training and fixed-init evaluation completed | [RLT results](../results/rlt_hf_spatial/summary.json) and [runbook](smolvla_rlt_spatial.md); progress-reward RLT 462/500 versus SFT 442/500 in one training seed |
| SmolVLA PPO via RLinf | Adapter and one 100-iteration Spatial run completed; external RLinf code path | [PPO experiment](smolvla_ppo_spatial.md), 500 fixed-init BF16 evaluation and portable per-episode records; multi-seed and same-runtime full SFT control remain open |
| ManiSkill3 / RoboCasa | Planned | Benchmark-specific adapters and documented equivalence boundaries, independent reproducible baseline |
| Agentic manipulation / RTC | Planned | High-level skill contract, verification/replanning tests, actual asynchronous latency/robustness study |

Future research should be added as measured recipes, not unimplemented feature flags. Keep raw pretrained-model weights and optional heavy dependencies outside the base installation. Introduce shared trainer abstractions only after at least two concrete algorithms expose the same behavior.

Possible resume wording **for the work currently measured**:

> Built a reproducible LIBERO-Spatial training/evaluation pipeline for multi-task ACT and Diffusion Policy, with fixed initial states, parallel rollout, execution-horizon ablations and per-episode provenance. Diagnosed checkpoint-selection mismatch: DP final improved SR from 69.4% to81.8%, approaching an83.0% ACT reference under matched evaluation.

Do not include π fine-tuning or RL achievements until those experiments are implemented and measured.
