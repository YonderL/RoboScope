#!/usr/bin/env bash
# Preview by default; --start trains and then runs the fixed-state evaluation.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/LIBERO/libero/libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/smolvla_spatial_seed0}"
RECIPE="${RECIPE:-${PROJECT_ROOT}/configs/libero_spatial/smolvla.json}"
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
    --output) OUTPUT_ROOT="${2:?--output requires a path}"; shift ;;
    --help|-h)
      echo "Usage: bash scripts/train_smolvla_spatial.sh [--start|--preview] [--resume] [--smoke] [--skip-eval] [--output DIR]"
      echo "Environment: PYTHON DATA_ROOT LIBERO_ROOT OUTPUT_ROOT RECIPE SMOLVLA_BASE_PATH SMOLVLA_VLM_PATH HF_ENDPOINT"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if ((smoke)); then OUTPUT_ROOT="${OUTPUT_ROOT}_smoke"; fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
resolved_recipe="$(mktemp "${TMPDIR:-/tmp}/roboscope-smolvla-recipe.XXXXXX")"
trap 'rm -f -- "$resolved_recipe"' EXIT
"$PYTHON" - "$RECIPE" "$resolved_recipe" "$PROJECT_ROOT" "$smoke" <<'PY'
import json
import os
import sys
from pathlib import Path

recipe, target, project, smoke = sys.argv[1:]
cfg = json.loads(Path(recipe).read_text())
cfg['hf_cache_dir'] = str(Path(project) / '.cache/smolvla/hub')
for variable, key in [('SMOLVLA_BASE_PATH', 'pretrained_path'), ('SMOLVLA_VLM_PATH', 'vlm_path')]:
    if os.environ.get(variable):
        cfg[key] = str(Path(os.environ[variable]).expanduser().resolve())
if int(smoke):
    cfg.update(train_steps=2, batch_size=2, micro_batch_size=1, workers=0,
               validate_every=1, validation_samples=2, save_every=1, log_every=1, eval_episodes=1)
Path(target).write_text(json.dumps(cfg, indent=2) + '\n')
PY
"$PYTHON" -m roboscope train --recipe "$resolved_recipe" \
  --data-root "$DATA_ROOT" --libero-root "$LIBERO_ROOT" --output "$OUTPUT_ROOT" \
  "${start_args[@]}" "${resume_args[@]}"
if ((skip_eval)); then exit 0; fi
episodes=50
if ((smoke)); then episodes=1; fi
"$PYTHON" -m roboscope evaluate --source "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/evaluation_final" --checkpoint final --episodes "$episodes" --ta 50 \
  "${start_args[@]}" "${resume_args[@]}"
