#!/usr/bin/env bash
# Run from any directory. Defaults match kty-ly; all paths can be overridden.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/liuyang/miniconda3/envs/lerobot/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/liuyang/robot_learning}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/LIBERO/libero/libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/pi0_lora_spatial_seed0}"
RECIPE="${RECIPE:-${PROJECT_ROOT}/configs/libero_spatial/pi0_lora.json}"
preview=0
smoke=0
skip_eval=0
resume_args=()
while (($#)); do
  case "$1" in
    --preview) preview=1 ;;
    --smoke) smoke=1 ;;
    --resume) resume_args=(--resume) ;;
    --skip-eval) skip_eval=1 ;;
    --output) OUTPUT_ROOT="${2:?--output requires a path}"; shift ;;
    --help|-h)
      echo "Usage: bash scripts/train_pi0_lora_spatial.sh [--preview] [--smoke] [--resume] [--skip-eval] [--output DIR]"
      echo "Environment: PYTHON DATA_ROOT LIBERO_ROOT OUTPUT_ROOT RECIPE PI0_BASE_PATH PI0_TOKENIZER_PATH HF_ENDPOINT"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON. Set PYTHON to the existing LeRobot 0.6.1 environment." >&2
  exit 1
fi
if ((smoke)); then OUTPUT_ROOT="${OUTPUT_ROOT}_smoke"; fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if ((!preview)) && ! "$PYTHON" -c 'import sentencepiece' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-deps -r "${PROJECT_ROOT}/requirements-pi0.txt"
fi
resolved_recipe="$(mktemp "${TMPDIR:-/tmp}/roboscope-pi0-recipe.XXXXXX")"
trap 'rm -f -- "$resolved_recipe"' EXIT
"$PYTHON" - "$RECIPE" "$resolved_recipe" "$PROJECT_ROOT" "$smoke" <<'PY'
import json, os, sys
from pathlib import Path
recipe, target, project, smoke = sys.argv[1:]
cfg = json.loads(Path(recipe).read_text())
cfg['hf_cache_dir'] = str(Path(project) / '.cache/pi0/hub')
for variable, key in [('PI0_BASE_PATH', 'pretrained_path'), ('PI0_TOKENIZER_PATH', 'tokenizer_path')]:
    if os.environ.get(variable):
        cfg[key] = str(Path(os.environ[variable]).expanduser().resolve())
if int(smoke):
    cfg.update(train_steps=2, warmup_steps=0, batch_size=2, micro_batch_size=1,
               gradient_accumulation_steps=1, workers=0, validate_every=1,
               validation_samples=2, save_every=1, log_every=1, eval_episodes=1)
Path(target).write_text(json.dumps(cfg, indent=2) + '\n')
PY
start_args=(--start)
if ((preview)); then start_args=(); fi
"$PYTHON" -m roboscope train --recipe "$resolved_recipe" \
  --data-root "$DATA_ROOT" --libero-root "$LIBERO_ROOT" --output "$OUTPUT_ROOT" \
  ${start_args[@]+"${start_args[@]}"} ${resume_args[@]+"${resume_args[@]}"}
if ((preview || skip_eval)); then exit 0; fi
episodes=50
if ((smoke)); then episodes=1; fi
"$PYTHON" -m roboscope evaluate --source "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/evaluation_final" --checkpoint final --episodes "$episodes" \
  --start ${resume_args[@]+"${resume_args[@]}"}
echo "Completed: ${OUTPUT_ROOT}/evaluation_final/summary.json"
