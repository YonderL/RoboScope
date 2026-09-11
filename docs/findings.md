# Research findings: checkpoint selection and closed-loop control

## Evidence, not a leaderboard claim

The matched study contains 500 fixed task/init pairs for each policy, with one training seed. ACT final achieves 415/500 (83.0%); DP final achieves 409/500 (81.8%). DP's earlier minimum-validation-loss checkpoint achieves 347/500 (69.4%). All DP main comparisons use DDIM=10, Ta=8.

![Checkpoint selection](assets/checkpoint_selection.png)

## Offline denoising loss selected a weaker controller

| Step | Online train noise MSE | EMA validation noise MSE | Rollout SR |
|---|---:|---:|---:|
| 7k | 0.0446 | **0.0465** | 69.4% |
| 30k | **0.0142** | 0.1405 | **81.8%** |

The labels `best` and `final` describe checkpoint rules, not policy quality. Validation averages epsilon error across sampled diffusion timesteps and action coordinates. It does not directly measure the consequences of gripper timing, contact errors or recovery. Online-train and EMA-validation curves also differ in model weights and crop mode, so their absolute gap is not a clean overfitting measure.

**Established here:** lower validation noise MSE did not rank these two checkpoints by closed-loop SR. **Not established:** why each failure occurred, whether every later checkpoint is better, or whether longer training would keep improving SR. Only 7k and 30k were evaluated in this checkpoint comparison.

Same-init pairing finds 93 rescued episodes and 31 lost episodes, net +62/500 = **+12.4 pp**. A seeded, task-stratified paired bootstrap gives **[8.4, 16.4] pp** (95%, 10,000 replicates). This interval conditions on these trained weights and resamples initial states within each task; it is not cross-training-seed uncertainty.

## ACT and DP final are close in aggregate, different by task

Moving from ACT to DP final rescues 42 episodes and loses 48: **−1.2 pp**, paired bootstrap **[−4.8, 2.4] pp**. This does not establish equivalence; it also does not support a stable ACT advantage from this single seed.

DP improves on “between plate/ramekin” and “next to ramekin”, but remains weak on “on ramekin” (4%, versus ACT 30%). Final improves drawer success from 50% to 90% and cabinet success from 48% to 84%. Aggregate SR alone hides these task-specific differences. A failure taxonomy should distinguish approach, grasp/contact, transport, release and recovery; those labels have not yet been annotated for all episodes.

## Inference settings are checkpoint-dependent

The existing DDIM 5/10/20 and Ta 1/4/8 sweeps were run on **7k only**. Ta=1 improved 7k SR to 76.0%, while more DDIM steps did not monotonically improve performance. These results must not be presented as the final checkpoint's ablations. Changing Ta also changes feedback frequency and compute cost.

Policy inference is synchronous and the simulator waits. DP's greater wall-clock inference latency does not itself cause simulated uncontrolled motion during prediction. A real-time asynchronous deployment would require a different experiment.

## Historical ACT study

![ACT ablation curves](assets/act_learning_curves.png)

Four chunk sizes × three seeds, 32 epochs, evaluations every 8 epochs. These curves use **20**, not 50, initial states per task. Shading is sample standard deviation across training seeds. TE performs much better than per-step unaggregated replanning for K=8/16, but offers no clear aggregate advantage over chunk execution. The latter comparison changes both aggregation and prediction frequency.

Positive TE coefficient 0.01 weights older predictions slightly more. This is a plausible source of stale predictions, especially for long chunks, but performance tables alone do not prove the mechanism. The exploratory gripper-exclusion test did not improve SR (85.5% vs rerun TE 89.0%), and baseline repetition varied from 87.0% to 89.0%. Dynamic batch numerical differences were measured; their full causal contribution to rollout divergence remains unresolved.

## Next experiments, in order

1. Establish repeatability under a fixed batch/scheduling protocol and inspect the persistent ramekin failure.
2. Evaluate final-checkpoint inference ablations without retraining.
3. Use a development rollout set for model/parameter selection and reserve separate fixed states for final reporting.
4. Add training seeds; distinguish per-state uncertainty from training variability.
5. Compare controlled visual representations and budgets before claiming architectural superiority.

The final-checkpoint test was added after observing the earlier score. Report it as exploratory; do not retrofit a claim of pre-registered checkpoint selection.
