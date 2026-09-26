# Recipes

Run from the repository root after following `docs/quickstart.md`.

- `configs/libero_spatial/act_baseline.json`: one K8 / seed0 model, 32 epochs, 50 episodes/task.
- `configs/libero_spatial/act_ablation.json`: K1/8/16/32 × seeds0/1/2, historical 20 episodes/task.
- `configs/libero_spatial/diffusion.json`: one suite-conditioned seed0 DP, 30k updates, batch128.
- `configs/libero_spatial/smolvla_official.json`: paper-architecture SmolVLA on official 256px Spatial data, 100k updates. The published 411/500 score executes 10 actions and is a different protocol from ACT/DP.
- `configs/libero_spatial/pi0_lora_hf_spatial.json`: Pi-0 LoRA on the HF Spatial dataset, 30k updates. The published record is the training curve, not a closed-loop score.

`roboscope train` trains; `roboscope evaluate` evaluates one checkpoint / DDIM / Ta combination. The historical ACT trainer also evaluates its execution modes at each specified epoch. DP evaluation is a separate command, allowing final and minimum-validation-loss checkpoints to be compared explicitly.

Examples of inference-only experiments (each command occupies both 4090s; run sequentially):

```bash
for steps in 5 10 20; do
  roboscope evaluate --source outputs/dp --checkpoint final --ddim-steps "$steps" --ta 8 \
    --output "outputs/dp_final_ddim${steps}_ta8" --start
done
for ta in 1 4; do
  roboscope evaluate --source outputs/dp --checkpoint final --ddim-steps 10 --ta "$ta" \
    --output "outputs/dp_final_ddim10_ta${ta}" --start
done
```

These are commands for future final-checkpoint ablations; they have not been run as part of the published results.
