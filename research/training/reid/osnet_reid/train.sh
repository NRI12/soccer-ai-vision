#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE="$SCRIPT_DIR/benchmarks/baseline"

CONFIG="${1:-configs/rtx3090_config.yaml}"
shift || true   # remaining args forwarded to main.py

echo "=== OSNet ReID - Train ==="
echo "Config : $CONFIG"
echo "Extra  : $*"
echo ""

cd "$BASELINE"
PYTHONUNBUFFERED=1 python3 main.py --config-file "$CONFIG" "$@"
