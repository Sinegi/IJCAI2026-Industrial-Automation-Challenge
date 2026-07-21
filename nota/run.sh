#!/usr/bin/env bash
# EQ / fsiq_csreg — QLoRA + CSReg pipeline wrapper.
#
# All model/dataset caches are redirected to /mnt/hdd3 because the home disk (`/`)
# is ~100% full. These env vars live ONLY for this process — no ~/.bashrc edits,
# no symlinks, nothing permanent (per the standing "config non-permanent" rule).
#
# Usage:
#   bash run.sh all                 # prepare -> train -> eval(val) -> submit(test)
#   bash run.sh prepare             # build train-only G* corpus on hdd3
#   bash run.sh train               # QLoRA + CSReg training
#   bash run.sh eval                # evaluate the saved adapter on val.jsonl
#   bash run.sh submit              # write submission.csv from test.jsonl
#   bash run.sh smoke               # tiny end-to-end sanity check (existing adapter, 20 rows)
#
# Extra flags pass straight through to the underlying script, e.g.:
#   bash run.sh train  --max-rows 200
#   bash run.sh eval   --adapter artifacts/csreg_r1_qwen7b/final_adapter --max-rows 20
#   CONFIG=configs/a100_40gb.yaml VAL=val.jsonl TEST=test.jsonl bash run.sh all
#   CUDA_VISIBLE_DEVICES=1 bash run.sh train
set -euo pipefail

CMD="${1:-all}"; shift || true

CONFIG="${CONFIG:-configs/a100_40gb.yaml}"
VAL="${VAL:-val.jsonl}"
TEST="${TEST:-test.jsonl}"

# --- storage on hdd3 (env vars scoped to this process only) ---
HDD_CACHE=/mnt/hdd3/user1/.cache
export HF_HOME="$HDD_CACHE/huggingface"
export HF_HUB_CACHE="$HDD_CACHE/huggingface/hub"
export HF_DATASETS_CACHE="$HDD_CACHE/huggingface/datasets"
export XDG_CACHE_HOME="$HDD_CACHE"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# --- runtime settings for this box ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --- activate the training env (python3 + torch/peft/bitsandbytes) ---
source /home/user1/anaconda3/etc/profile.d/conda.sh
conda activate eq_vllm
cd "$(dirname "$0")"

echo "[run.sh] cmd=$CMD  config=$CONFIG"
echo "[run.sh] HF_HOME=$HF_HOME"
echo "[run.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  python=$(python -V 2>&1)"

run_prepare() { python scripts/prepare_data.py --config "$CONFIG" "$@"; }
run_train()   { python scripts/train.py        --config "$CONFIG" "$@"; }
run_eval()    { python scripts/evaluate.py      --config "$CONFIG" --val  "$VAL"  --output-dir outputs/validation "$@"; }
run_submit()  { python scripts/submit.py        --config "$CONFIG" --test "$TEST" --output submission.csv "$@"; }

case "$CMD" in
  prepare) run_prepare "$@" ;;
  train)   run_train   "$@" ;;
  eval)    run_eval    "$@" ;;
  submit)  run_submit  "$@" ;;
  all)
    run_prepare
    run_train
    run_eval
    run_submit
    ;;
  smoke)
    # No training / no prepare: exercise eval+submit on the existing adapter over a few rows.
    run_eval   --max-rows 20 --output-dir outputs/validation_smoke "$@"
    python scripts/submit.py --config "$CONFIG" --test "$TEST" \
      --output outputs/smoke_submission.csv --max-rows 20 "$@"
    ;;
  *)
    echo "unknown command: $CMD" >&2
    echo "expected one of: all | prepare | train | eval | submit | smoke" >&2
    exit 2
    ;;
esac
