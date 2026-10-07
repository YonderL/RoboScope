# Validation record

## Review and follow-up, 2026-09-26

- **102 tests passed** in the existing PyTorch/LeRobot environment, with CUDA
  hidden. This includes the two-process CPU checkpoint/resume test. Two upstream
  SWIG deprecation warnings remain.
- **56 tests passed, 11 skipped** in a separate CPU reporting environment without
  PyTorch or LeRobot. Optional model tests skip cleanly; reporting, previews,
  result-integrity checks and source-release checks run without the training stack.
- Ruff lint and formatting checks pass. A wheel was built without fetching or
  upgrading training dependencies, installed into an isolated directory, and its
  CLI imported from that installation.
- Fixed a task-subset aggregation bug: an empty even-task shard omitted its
  `episodes.jsonl`, causing a completed odd-task evaluation to fail aggregation.
  Regression tests retain strict failure for missing nonempty shards.
- Published native results reject duplicate/missing tasks, incorrect denominators,
  aggregate mismatches and non-boolean outcomes. Follow-up records additionally
  validate checkpoint identity, fixed-state seeds and episode identity conventions.
- The release builder rejects symlinks, model/data artifacts and personal absolute
  paths, including in shell scripts. Local hardware probes, temporary continuation
  notes, raw runs and duplicate PDF/SVG exports are excluded.

The [follow-up evaluation report](evaluation_20260926.md) gives real PRO 5000 /
MuJoCo 3.3.2 rollout results separately from these engineering checks. Pi-0's
10-episode smoke test (8 successes) is local verification only; it is not pooled
into the formal 500-episode study.

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest tests
python -m roboscope report --study all
python scripts/build_release.py
```

## Historical migration verification, 2026-09-11

The original migration audited 3,500 ACT/DP episode records and recomputed
ACT 415/500, DP 7k 347/500 and DP 30k 409/500. Fourteen bounded CPU tests passed.
For a fixed batch of two inputs, the archived and packaged ACT/DP implementations
had zero maximum absolute prediction difference in BF16 on RTX 4090. Each policy
also completed two simulator episodes capped at three steps; these were smoke
checks, not benchmark trials. The old machine-specific diagnostic file is local.

```bash
# Optional local comparison against the original archived run.
PYTHONPATH=src python scripts/smoke_migration.py \
  --archive outputs/dp_spatial_seed0 --report /tmp/migration_smoke.json
```

## Limits

Full ACT/DP training and all of their historical suite evaluations were not
repeated after refactoring. The new on-ramekin trials are explicitly separate.
Single-seed success rates do not establish cross-seed robustness, and changes
across GPU, PyTorch, simulator or batching settings need not be bitwise identical.
Clean-machine provisioning of every training dependency remains unverified.
RLT has implementation and bounded tests, but no published success-rate gain.
