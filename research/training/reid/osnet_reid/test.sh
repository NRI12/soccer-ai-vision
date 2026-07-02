#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE="$SCRIPT_DIR/benchmarks/baseline"

if [ -z "$1" ]; then
    echo "Usage: bash test.sh <checkpoint.pth> [save_dir] [extra args...]"
    echo "  checkpoint.pth  path to model weights"
    echo "  save_dir        where to write logs/rankings (default: logs/eval)"
    exit 1
fi

CHECKPOINT="$1"
SAVE_DIR="${2:-logs/eval}"
shift 2 || shift 1 || true

echo "=== OSNet ReID - Test ==="
echo "Checkpoint : $CHECKPOINT"
echo "Save dir   : $SAVE_DIR"
echo ""

cd "$BASELINE"
PYTHONUNBUFFERED=1 python3 main.py \
    --config-file configs/rtx3090_config.yaml \
    test.evaluate True \
    model.load_weights "$CHECKPOINT" \
    data.save_dir "$SAVE_DIR" \
    "$@"
