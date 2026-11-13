#!/usr/bin/env bash
# Rebuild every table and figure in results/ from scratch (~35 min on 4 cores).
# Set TRACK=1 to also log everything to MLflow (needs requirements-tracking.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
T=${TRACK:+--track}
python -m src.data                      # download + clean -> data/processed/
python scripts/01_eda.py                # Phase 1 figures + results/eda_summary.md
python scripts/01_eda.py --dataset suzuki_miyaura
python scripts/02_baseline_model.py $T  # Phase 2 results/baseline_metrics.md
python -m src.benchmark --suite main --seeds 50 --budget 100 $T
for suite in encodings batch acquisition noise; do
  python -m src.benchmark --suite "$suite" --seeds 30 --budget 100 $T
done
python -m src.benchmark --dataset suzuki_miyaura --suite main --seeds 50 --budget 100 $T
python -m src.benchmark --dataset suzuki_miyaura --suite encodings --seeds 30 --budget 100 $T
