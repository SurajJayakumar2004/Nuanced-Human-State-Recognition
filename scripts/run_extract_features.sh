#!/usr/bin/env bash
# Run feature extraction with MPS fallback enabled (macOS).
# Usage: ./scripts/run_extract_features.sh   or   bash scripts/run_extract_features.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec python3 -m scripts.extract_features "$@"
