# SmolVLA PPO on LIBERO-Spatial through RLinf

This experiment starts from the official Spatial SmolVLA SFT 100k checkpoint and trains the RLinf SmolVLA adapter for 100 rollout/update iterations. The adapter source belongs to the separate RLinf repository; RoboScope publishes the portable Spatial override and compact evaluation evidence. It is distinct from the frozen-VLA RLT experiment and from the four-policy official comparison.

## Reproduce the training configuration

Use the SmolVLA-enabled RLinf branch and copy [the Spatial override](../configs/rlinf/smolvla_ppo_spatial.yaml) into its `examples/embodiment/config/` directory. Set `SMOLVLA_SFT_COMPAT_DIR` to a LeRobot 0.4.4-compatible copy of the official SFT 100k checkpoint, and `SMOLVLA_PPO_OUTPUT_DIR` to a writable output directory. The original SFT weights and processors can be referenced by symlinks; the compatibility copy only omits the `pretrained_revision: null` field that LeRobot 0.4.4 cannot parse. Set `REPO_PATH`, `EMBODIED_PATH`, `ROBOT_PLATFORM=LIBERO`, and the simulator GPU/EGL binding for the local machine before running RLinf's `examples/embodiment/run_embodiment.sh smolvla_ppo_spatial`. The RLinf adapter's generic example is for Show-Harness LIBERO-10 (9D joints, 128px OpenCV cameras); the published override changes this to official HF Spatial (8D EEF state, 256px images, OpenGL rotation, `image`/`image2` cameras).

The Spatial run used one RTX PRO 5000 GPU with 20 parallel training environments. Its key parameters were:

| Setting | Value |
|---|---:|
| Rollout/update iterations | 100 |
| Maximum controlled steps per episode | 280 |
| Action prediction / execution horizon | 50 / 10 |
| Flow steps / sampled transition noise level | 10 / 0.5 |
| PPO epochs; global / micro batch | 2; 40 / 2 |
| Actor / value learning rate | 1e-7 / 1e-4 |
| AdamW betas, epsilon, weight decay | (0.9, 0.95), 1e-8, 0.01 |
| Ratio clip; dual clip | 0.2; 3.0 |
| Value clip; Huber delta | 0.2; 10 |
| GAE gamma / lambda | 0.99 / 0.95, at action-chunk level |
| Gradient clipping | 1.0 |
| Entropy bonus / reference KL | 0 / 0 |
| Actor seed / environment seed | 42 / 0 |

The model wrapper used root FSDP `no_shard`, FP32 model parameters, `use_orig_params=true`, and no gradient checkpointing. The actual rollout and evaluation inference paths used BF16. Training reward was the LIBERO success signal and relative reward, without a separate reward model or SFT mixing loss. Each block randomly selected one denoising transition for its Gaussian log-prob; the other flow steps used ODE integration. The objective applies PPO to that transition's executed action coordinates, not the exact likelihood of the final environment action. The value head used the observation prefix and detached its critic input. A requested critic warmup of 28 steps failed with the current FSDP configuration; the completed run used `critic_warmup_steps=0`.

## Observed outcomes

All 100 iterations completed and the step-100 checkpoint was saved. The 50-episode in-training evaluation ran every ten iterations; its final result was **40/50 (80%)**, the same as the SFT baseline under that training-time protocol. Intermediate results ranged from 36/50 to 40/50. These checks are a trend monitor, not the fixed-initial-state 500-episode evaluation. The [training curve](../results/smolvla_ppo_spatial/training_eval.csv) preserves every check.

The final BF16 evaluation used 10 LIBERO-Spatial tasks with initial-state IDs 0–49 for each task, MuJoCo 3.3.2, hf-libero 0.1.4, robosuite 1.4.0, the corrected `property_sampler_clear_v1` reset protocol, 10-step ODE sampling, and execution of 10 actions per model call. The model ran with LeRobot 0.4.4 and Transformers 4.57.6. The PPO checkpoint succeeded in **449/500 (89.8%)**, compared with **442/500 (88.4%)** for the historical native SFT control and **462/500 (92.4%)** for historical progress-reward RLT. Its 500 physical initial-state fingerprints matched the archived RLT evaluation. The [summary](../results/smolvla_ppo_spatial/summary.json) and [500 episode records](../results/smolvla_ppo_spatial/episodes.csv) contain task IDs, outcomes and reset fingerprints. The retained step-100 actor file has SHA256 `61aa2582f17165768c653191ad4c7095bab1a660dda7658f24e71069b5026b3c`; the weights are not in Git.

| Native task ID | Historical SFT | PPO100 BF16 | Progress RLT |
|---:|---:|---:|---:|
| 0 | 48/50 | 49/50 | 48/50 |
| 1 | 46/50 | 47/50 | 47/50 |
| 2 | 49/50 | 49/50 | 49/50 |
| 3 | 44/50 | 42/50 | 49/50 |
| 4 (top drawer) | 41/50 | 43/50 | 45/50 |
| 5 | 42/50 | 45/50 | 44/50 |
| 6 | 50/50 | 49/50 | 50/50 |
| 7 | 41/50 | 43/50 | 45/50 |
| 8 | 39/50 | 38/50 | 46/50 |
| 9 | 42/50 | 44/50 | 39/50 |
| **All tasks** | **442/500** | **449/500** | **462/500** |

Task 4 also has a same-BF16-model-runtime SFT check: 42/50 versus PPO100's 43/50. The historical SFT and RLT model paths used LeRobot 0.6.1 and Transformers 5.5.4, while PPO used the RLinf adapter's versions above. The observed **+7/500 (+1.4 percentage points)** versus historical SFT is a small descriptive gain from one training seed. A full same-runtime SFT evaluation and multiple training seeds would establish its stability more clearly. An earlier FP32 diagnostic script produced erroneous task-4 scores; actual PPO rollouts and the final evaluation used BF16, so those FP32 scores are excluded.

Evaluation uses the normal ODE sampler after drawing the initial Gaussian latent. That is the intended inference path; the stochastic one-transition sampler is needed to compute the PPO training objective. The distinction limits how directly the surrogate loss predicts final closed-loop success.
