# Migration from experiment scripts

| Original implementation | Packaged implementation |
|---|---|
| ACT `data.py`, DP `act_reference/data.py` | `roboscope.data.libero` |
| DP `dataset.py` | `roboscope.data.sequences` |
| ACT `model.py` | `roboscope.policies.act` |
| DP `policy.py` | `roboscope.policies.diffusion` |
| ACT `run.py` training | `roboscope.trainers.act` |
| DP `train.py` | `roboscope.trainers.diffusion` |
| Simulator setup and environment processes | `roboscope.envs.libero`, `roboscope.envs.pool` |
| ACT parallel execution ablation | `roboscope.evaluation.act` |
| DP matched evaluation | `roboscope.evaluation.worker` |
| Launchers | `roboscope.workflows.experiments`, `roboscope.runtime.gpu` |
| One-off comparison | `roboscope.reporting.records`, `roboscope.reporting.figures` |

Old local `experiments/`, `outputs/` and `LIBERO/` directories are retained and excluded from source releases. Old run snapshots are the reference for historical results. Do not resume an archived experiment with reorganized source under its original output directory. The new evaluation workflow can read an existing checkpoint into a **new** evaluation directory; the archival standalone script remains the baseline for numerical regression.

Scientific recipes live in `configs/libero_spatial/`; local data/assets/output paths come from CLI flags. New DP training no longer requires a finished ACT run. The deterministic split and statistics are prepared through the shared data adapter.

The public source is `src/roboscope`, not a wrapper around ignored legacy folders. Formal experiment configurations remain explicit; the two evaluator schedules (historical ACT vs matched ACT–DP) are kept distinct to avoid silently changing baseline behavior.
