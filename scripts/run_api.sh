#!/bin/bash
# Start FastAPI ALPR Server
# Usage: bash scripts/run_api.sh

echo "=== Starting ALPR API Server ==="
python -m src api --host 0.0.0.0 --port 8000 "$@"
