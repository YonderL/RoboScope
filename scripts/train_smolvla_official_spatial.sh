#!/usr/bin/env bash
# Independent paper-based experiment. Preview does not download or touch CUDA.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/datasets/libero_official}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/libero_official_spatial}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache/smolvla_official}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/LIBERO/libero/libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/smolvla_official_spatial_seed0}"
RECIPE="${RECIPE:-${PROJECT_ROOT}/configs/libero_spatial/smolvla_official.json}"
stage=train
start_args=()
resume_args=()
skip_eval=0
while (($#)); do
  case "$1" in
    --stage) stage="${2:?--stage requires prepare, smoke, train, or evaluate}"; shift ;;
    --start) start_args=(--start) ;;
    --preview) start_args=() ;;
    --resume) resume_args=(--resume) ;;
    --skip-eval) skip_eval=1 ;;
    --output) OUTPUT_ROOT="${2:?--output requires a path}"; shift ;;
    --help|-h)
      echo "Usage: bash scripts/train_smolvla_official_spatial.sh [--stage prepare|smoke|train|evaluate] [--start|--preview] [--resume] [--skip-eval] [--output DIR]"
      echo "Environment: PYTHON SOURCE_ROOT DATA_ROOT CACHE_ROOT LIBERO_ROOT OUTPUT_ROOT RECIPE"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$stage" in prepare|smoke|train|evaluate) ;; *) echo "Invalid stage: $stage" >&2; exit 2 ;; esac
if [[ "$stage" == prepare || "$stage" == evaluate ]] && ((${#resume_args[@]})); then
  echo "--resume is only valid with train or smoke" >&2
  exit 2
fi
if [[ "$stage" == smoke ]]; then OUTPUT_ROOT="${OUTPUT_ROOT}_smoke"; fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
if [[ "$stage" == prepare ]] && ((${#start_args[@]})); then
  "$PYTHON" -m roboscope.data.smolvla_official \
    --source "$SOURCE_ROOT" --destination "$DATA_ROOT" --libero-root "$LIBERO_ROOT"
fi
common=(--recipe "$RECIPE" --dataset "$DATA_ROOT" --cache "$CACHE_ROOT"
        --libero-root "$LIBERO_ROOT" --output "$OUTPUT_ROOT")
"$PYTHON" -m roboscope.workflows.smolvla_official \
  "${common[@]}" --stage "$stage" "${start_args[@]}" "${resume_args[@]}"
if [[ "$stage" == train ]] && ((skip_eval == 0)); then
  "$PYTHON" -m roboscope.workflows.smolvla_official \
    "${common[@]}" --stage evaluate "${start_args[@]}"
fi
