#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_DIR="$HOME/.venvs/cot-overthinking-bede"
VENV_DIR="${1:-$DEFAULT_VENV_DIR}"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$VENV_DIR"
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

mkdir -p "$HF_HOME_DIR"

cat <<EOF
Environment created at:
  $VENV_DIR

Hugging Face cache directory:
  $HF_HOME_DIR

Next time, activate it with:
  source $VENV_DIR/bin/activate
EOF
