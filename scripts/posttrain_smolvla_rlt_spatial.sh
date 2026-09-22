#!/usr/bin/env bash
# RLT from an existing SFT checkpoint. No training/evaluation without --start.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SFT_SOURCE="${SFT_SOURCE:-${PROJECT_ROOT}/outputs/smolvla_spatial_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/smolvla_rlt_spatial_seed0}"
RECIPE="${RECIPE:-${PROJECT_ROOT}/configs/libero_spatial/smolvla_rlt.json}"
checkpoint=final
stage=all
start_args=()
resume_args=()
smoke=0
skip_eval=0
while (($#)); do
  case "$1" in
    --start) start_args=(--start) ;;
    --preview) start_args=() ;;
    --resume) resume_args=(--resume) ;;
    --smoke) smoke=1 ;;
    --skip-eval) skip_eval=1 ;;
    --source) SFT_SOURCE="${2:?--source requires a path}"; shift ;;
    --output) OUTPUT_ROOT="${2:?--output requires a path}"; shift ;;
    --checkpoint) checkpoint="${2:?--checkpoint requires best or final}"; shift ;;
    --stage) stage="${2:?--stage requires all, token, warmup, online or evaluate}"; shift ;;
    --help|-h)
      echo "Usage: bash scripts/posttrain_smolvla_rlt_spatial.sh [--start|--preview] [--stage all|token|warmup|online|evaluate] [--source SFT_RUN] [--output DIR] [--checkpoint best|final] [--resume] [--smoke] [--skip-eval]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$stage" in
  all|token|warmup|online|evaluate) ;;
  *) echo "Invalid stage: $stage" >&2; exit 2 ;;
esac
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
resolved_recipe="$(mktemp "${TMPDIR:-/tmp}/roboscope-rlt-recipe.XXXXXX")"
trap 'rm -f -- "$resolved_recipe"' EXIT
"$PYTHON" - "$RECIPE" "$resolved_recipe" "$smoke" <<'PY'
import json
import sys
from pathlib import Path

recipe, target, smoke = sys.argv[1:]
cfg = json.loads(Path(recipe).read_text())
if int(smoke):
    cfg.update(token_steps=2, token_batch_size=2, feature_batch_size=1, token_save_every=1,
               online_steps=120, warmup_steps=20, initial_updates=2, updates_per_transition=1,
               batch_size=4, replay_capacity=100, rollout_horizon=20, workers=0,
               log_every=1, eval_episodes=1, eval_envs=1, video_episodes_per_task=0)
Path(target).write_text(json.dumps(cfg, indent=2) + '\n')
PY
episodes=50
if ((smoke)); then OUTPUT_ROOT="${OUTPUT_ROOT}_smoke"; episodes=1; fi
if [[ "$stage" != evaluate ]]; then
  "$PYTHON" -m roboscope posttrain --source "$SFT_SOURCE" --recipe "$resolved_recipe" \
    --output "$OUTPUT_ROOT" --checkpoint "$checkpoint" --stage "$stage" "${start_args[@]}" "${resume_args[@]}"
fi
if [[ "$stage" == token || "$stage" == warmup ]]; then exit 0; fi
if ((skip_eval)); then exit 0; fi
"$PYTHON" -m roboscope evaluate --source "$OUTPUT_ROOT" --output "${OUTPUT_ROOT}/evaluation_rlt" \
  --episodes "$episodes" "${start_args[@]}" "${resume_args[@]}"
"$PYTHON" -m roboscope evaluate --source "$OUTPUT_ROOT" --output "${OUTPUT_ROOT}/evaluation_sft_c10" \
  --episodes "$episodes" --rlt-reference "${start_args[@]}" "${resume_args[@]}"
