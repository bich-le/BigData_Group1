#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-classification}"
case "$TASK" in
  classification|regression) ;;
  *) echo "Usage: $0 classification|regression" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_MEMORY="${DRIVER_MEMORY:-32g}"
INPUT_DIR="/tmp/flight-${TASK}-input"
OUTPUT_DIR="/tmp/flight-${TASK}-output"
CHECKPOINT_DIR="/tmp/flight-${TASK}-checkpoint"

rm -rf "$INPUT_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
mkdir -p "$INPUT_DIR"
cp "$ROOT/docs/sample_${TASK}_stream_input.json" "$INPUT_DIR/events.json"

spark-submit --master 'local[2]' --driver-memory "$DRIVER_MEMORY" --conf spark.driver.maxResultSize=4g \
  "$ROOT/validate_spark_model.py" --task "$TASK"

spark-submit --master 'local[2]' --driver-memory "$DRIVER_MEMORY" --conf spark.driver.maxResultSize=4g \
  "$ROOT/inference.py" --task "$TASK" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --available-now

cat "$OUTPUT_DIR"/*.json
