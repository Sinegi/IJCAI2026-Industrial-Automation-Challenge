#!/usr/bin/env bash
# Official starter kit + the EQ / fsiq_csreg model.
#
# Everything needed lives in this directory: inference code (fsiq_csreg/), config
# (configs/eq_nota_gen.yaml), trained adapter (adapter/), data (data/). The only
# download is the base model, Qwen/Qwen3-8B from the HuggingFace hub.
#
# Usage:
#   bash run_eq.sh check                 # resolve config/adapter only -- no model load, no GPU
#   bash run_eq.sh smoke                 # first 20 test rows, end to end
#   bash run_eq.sh test                  # full run -> competition_results/submission.csv
#   bash run_eq.sh val                   # val.jsonl -> competition_results/submission_val.csv
#
# Overrides:
#   CUDA_VISIBLE_DEVICES=1 EQ_BATCH_SIZE=64 bash run_eq.sh test
#   EQ_ADAPTER=none bash run_eq.sh smoke              # base model, no adapter
#   EQ_CONDA_ENV=eq_vllm bash run_eq.sh test          # activate a conda env first
#   bash run_eq.sh test --dataset-path /path/to/kaggle_file.jsonl
set -euo pipefail

CMD="${1:-test}"; shift || true

KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$KIT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Optional: activate a conda env holding torch/peft/bitsandbytes. Without
# EQ_CONDA_ENV the current interpreter is used, so a plain venv works too.
if [ -n "${EQ_CONDA_ENV:-}" ]; then
  CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$EQ_CONDA_ENV"
  else
    echo "[run_eq.sh] conda not found; using $(command -v python)" >&2
  fi
fi
# Default to python3: a bare `python` is still Python 2 on some machines.
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "[run_eq.sh] need Python >= 3.10, got $($PYTHON -V 2>&1) at $PYTHON" >&2
  echo "[run_eq.sh] set PYTHON=/path/to/python or EQ_CONDA_ENV=<env>" >&2
  exit 2
}

case "$CMD" in
  check|smoke|test) CONFIG_JSON="metadata_config_test.json" ;;
  val)              CONFIG_JSON="metadata_config_val.json" ;;
  *) echo "unknown command: $CMD (expected check | smoke | test | val)" >&2; exit 2 ;;
esac

read_key() { "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))$2)" "$1"; }

DATASET="$(read_key "$CONFIG_JSON" "['dataset']['dataset_path']")"
OUT_DIR="$(read_key "$CONFIG_JSON" ".get('output_dir','competition_results')")"
OUT_FILE="$(read_key "$CONFIG_JSON" ".get('output_file','submission.csv')")"

# The starter kit hands the predictor one scenario at a time and never tells it
# which file they came from; EQ_DATASET is how the predictor learns to warm-start
# the whole file in batches instead.
export EQ_DATASET="$DATASET"

echo "[run_eq.sh] cmd=$CMD  kit_config=$CONFIG_JSON  dataset=$DATASET"
echo "[run_eq.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  python=$($PYTHON -V 2>&1)"
"$PYTHON" eq_predictor.py

if [ "$CMD" = "check" ]; then
  exit 0
fi

if [ "$CMD" = "smoke" ]; then
  # 20 rows only; warm-start over the subset so the smoke run stays cheap.
  SUBSET="$OUT_DIR/_smoke_subset.jsonl"
  "$PYTHON" - "$DATASET" "$SUBSET" <<'PY'
import sys
from pathlib import Path
source, target = Path(sys.argv[1]), Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
lines = source.read_text(encoding="utf-8").splitlines()[:20]
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[run_eq.sh] smoke subset: {target} ({len(lines)} rows)")
PY
  export EQ_DATASET="$SUBSET"
  OUT_FILE="smoke_${OUT_FILE}"
  "$PYTHON" run.py --config "$CONFIG_JSON" --dataset-path "$SUBSET" --output-file "$OUT_FILE" -v "$@"
else
  "$PYTHON" run.py --config "$CONFIG_JSON" -v "$@"
fi

# eval_framework.py catches predictor exceptions and writes NOTAVALUE, so a run
# with a dead model still exits 0. Fail loudly instead.
"$PYTHON" - "$OUT_DIR/$OUT_FILE" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(encoding="utf-8")))
if not rows:
    raise SystemExit(f"{path} is empty")
if list(rows[0]) != ["id", "answer"]:
    raise SystemExit(f"columns must be exactly id,answer -- got {list(rows[0])}")
ids = [r["id"] for r in rows]
if len(set(ids)) != len(ids):
    raise SystemExit("duplicate ids in submission")
bad = [r["id"] for r in rows if not r["answer"] or r["answer"] == "NOTAVALUE"]
if bad:
    raise SystemExit(f"{len(bad)} rows have no answer (e.g. {bad[:5]}) -- the predictor errored")
print(f"[run_eq.sh] OK: {len(rows)} rows -> {path}")
PY
