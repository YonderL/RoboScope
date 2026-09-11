# Validation record

Validated locally on 2026-09-11. These are bounded engineering checks, not new benchmark scores.

## Completed

- Audited all **3,500** published episodes: exact task/init coverage, no duplicates, binary success, matching episode seeds and checkpoint identity across both shards.
- Recomputed SR directly from records: ACT415/500; DP7k347/500; DP30k409/500. Regenerated PNG/PDF/SVG charts and paired, task-stratified bootstrap summaries.
- **14 CPU tests passed**, including a repeat from the extracted standalone source archive. They cover record corruption/incompleteness, preview commands without data/GPU access, worker entrypoint imports, rejected unsupported horizons, source-change rejection on resume, legacy path migration, temporal dataset boundaries, deterministic sampler resume, independent GN CNNs, task-embedding gradients, center cropping, deterministic DDIM and TE target-time alignment.
- **Two RTX4090s only:** compared original archived model implementations with packaged models, using existing ACT epoch32 and DP30k checkpoints and identical batch2 inputs. BF16 prediction maximum absolute difference: **0.0 for both models**.
- **Two RTX4090s only:** each policy ran two simulator episodes with a maximum of three steps per episode. Independent process startup, observations, action execution, video encoding and trajectory output passed. These episodes are not added to experimental results.
- CPU figure regeneration from the extracted source archive passed. A wheel was built without resolving/upgrading dependencies, installed into an isolated temporary directory, and its CLI loaded successfully. Source-release export and all documentation-link checks passed.

Machine-readable GPU diagnostic: [migration_smoke.json](assets/migration_smoke.json).

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest tests
python -m roboscope report
python scripts/build_release.py
# Optional, bounded; requires your original local run and exactly two RTX4090s.
PYTHONPATH=src python scripts/smoke_migration.py \
  --archive outputs/dp_spatial_seed0 --report /tmp/migration_smoke.json
```

## Not established by these checks

- A complete ACT/DP training run and all 500-episode evaluations were **not** repeated after refactoring.
- Cross-batch bitwise equivalence, cross-machine reproducibility and clean-room provisioning of all simulator dependencies were not established.
- No π-series, RL post-training, alternative benchmark or asynchronous inference has been tested or implemented.
- GPU smoke comparisons use a small fixed batch. Exact agreement on that fixture does not imply every rollout or future dependency version is identical.

The published metrics come from the archived experiments; package migration tests establish a useful engineering baseline without relabeling old results as a new reproduction.
