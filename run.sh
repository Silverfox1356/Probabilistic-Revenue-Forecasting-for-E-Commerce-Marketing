#!/usr/bin/env bash
set -euo pipefail

# Positional args with local-run defaults (see Hackathon Submission Guide).
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"
FEATURES="$OUTPUT_DIR/features.csv"

# 1. Normalize the raw channel CSVs into the feature table the model expects.
python generate_features.py --data-dir "$DATA_DIR" --out "$FEATURES"

# 2. Load the pickled model and write probabilistic forecasts.
python predict.py --features "$FEATURES" --model "$MODEL_PATH" --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
