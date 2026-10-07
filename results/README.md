# Published evidence

| Directory | Evidence | Regenerate figures |
|---|---|---|
| `official_spatial/` | Official four-policy, ten-task SR table; corrected on-ramekin results merged with archived tasks | `python scripts/export_official_results.py` then `roboscope report --study official` |
| `libero_spatial/` | Historical ACT/DP suite comparison, checkpoint selection and execution ablations | `roboscope report --study baseline` |
| `vla_spatial/` | Native SmolVLA 100k/500-episode result; SmolVLA and Pi-0 training curves | `roboscope report --study vla` |
| `mujoco332/` | PRO 5000 follow-up: ACT/DP/SmolVLA on-ramekin, Pi-0 final over all ten tasks | `roboscope report --study mujoco332` |
| `rlt_hf_spatial/` | SmolVLA SFT C=10, sparse RLT, and progress-reward RLT fixed-initial-state comparison | `summary.json` is the portable per-task record |

Each study keeps its own protocol. A native episode position is not relabeled as
a fixed initial-state ID. Historical rows remain archived. The official selection
replaces only on-ramekin for ACT/DP/SmolVLA and uses the complete Pi-0 LoRA run. Source checksums and checkpoint identities accompany the retained records.

Training recipes live in `configs/libero_spatial/`. Follow-up evaluation settings,
environment versions and benchmark fingerprints are in `mujoco332/protocol.json`;
the commands are in [the evaluation report](../docs/evaluation_20260926.md).
The RLT comparison is a compact derived record; its training configuration is
`configs/libero_spatial/smolvla_rlt_hf_progress.json`.
The exporter requires complete, disjoint episode coverage before publishing.

Only lightweight scientific records are committed. Checkpoints, datasets,
videos, simulator traces, raw logs, smoke runs and superseded diagnostics remain
local. PNG previews are committed under `docs/assets/`; PDF/SVG exports can be
regenerated when needed.
