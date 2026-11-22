#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-classification}"
case "$TASK" in
  classification|regression) ;;
  *) echo "Usage: $0 classification|regression" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_MEMORY="${DRIVER_MEMORY:-4g}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/${TASK}_full_pipeline}"
SPARK_SUBMIT="$(command -v spark-submit || true)"
PYSPARK_RUNTIME_PYTHON="${PYSPARK_PYTHON:-}"
if [[ -z "$SPARK_SUBMIT" ]]; then
  for python_bin in "${VIRTUAL_ENV:-}/bin/python" "$ROOT/.venv/bin/python" "$ROOT/../.venv/bin/python"; do
    [[ -x "$python_bin" ]] || continue
    candidate="$("$python_bin" -c 'import os, pyspark; print(os.path.join(os.path.dirname(pyspark.__file__), "bin", "spark-submit"))' 2>/dev/null || true)"
    if [[ -x "$candidate" ]]; then
      SPARK_SUBMIT="$candidate"
      PYSPARK_RUNTIME_PYTHON="$python_bin"
      break
    fi
  done
fi
if [[ -z "$SPARK_SUBMIT" ]]; then
  echo "spark-submit was not found on PATH or in the project virtual environment." >&2
  exit 1
fi
if [[ -n "$PYSPARK_RUNTIME_PYTHON" ]]; then
  export PYSPARK_PYTHON="$PYSPARK_RUNTIME_PYTHON"
fi
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/flight-${TASK}-XXXXXX")"
INPUT_DIR="$RUN_DIR/input"
OUTPUT_DIR="$RUN_DIR/output"
CHECKPOINT_DIR="$RUN_DIR/checkpoint"
trap 'rm -rf "$RUN_DIR"' EXIT

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Missing full PipelineModel: $MODEL_DIR" >&2
  echo "Train one first with mllib_pipeline.py; existing legacy two-stage artifacts are intentionally rejected." >&2
  exit 1
fi
mkdir -p "$INPUT_DIR"
cp "$ROOT/docs/contracts/sample_${TASK}_stream_input.json" "$INPUT_DIR/events.json"

"$SPARK_SUBMIT" --master 'local[2]' --driver-memory "$DRIVER_MEMORY" \
  "$ROOT/inference.py" --task "$TASK" \
  --model "$MODEL_DIR" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --available-now

find "$OUTPUT_DIR" -name '*.json' -type f -exec cat {} +
