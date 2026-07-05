#!/bin/bash
# Wrapper: activate conda PatternRecognition environment and execute the given command.
# Usage: bash scripts/run.sh python script.py [args...]
#        bash scripts/run.sh conda list

CONDA_SH="/c/Developer/Anaconda/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate PatternRecognition
else
    echo "ERROR: conda.sh not found at $CONDA_SH" >&2
    exit 1
fi

exec "$@"
