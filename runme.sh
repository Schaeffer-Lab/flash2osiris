#!/usr/bin/env bash
# Generate an OSIRIS deck from a run.yaml (the single source of truth).
# Usage:  ./runme.sh [path/to/run.yaml]   (defaults to the 1D example)
# Run from the repo root; outputs land in ./input_files/<inputfile_name>/.
set -euo pipefail

CONFIG="${1:-examples/perlmutter_1d.run.yaml}"

# conda activate flash2osiris   # uncomment once the env is created (see environment.yml)

python -m flash_osiris.generator --config "$CONFIG"
