#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_DIR="$HOME/.venvs/cot-overthinking-lambda"
VENV_DIR="${1:-$DEFAULT_VENV_DIR}"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv --system-site-packages "$VENV_DIR"

source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade-strategy only-if-needed -r "$SCRIPT_DIR/requirements.txt"

mkdir -p "$HF_HOME_DIR" "$HF_HOME_DIR/hub" "$HF_HOME_DIR/transformers" "$HF_HOME_DIR/datasets"

python3 - <<'PY'
import platform

print(f"Python version: {platform.python_version()}")

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch was not importable inside the virtual environment. "
        "These Lambda instructions assume the default Lambda Stack image, "
        "where PyTorch is preinstalled system-wide and exposed through "
        "`python3 -m venv --system-site-packages`."
    ) from exc

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch path: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as exc:
    raise SystemExit("Transformers import failed after installing requirements.") from exc
PY

cat <<EOF
Lambda environment created at:
  $VENV_DIR

Hugging Face cache root:
  $HF_HOME_DIR

This setup uses:
  python3 -m venv --system-site-packages

That lets the venv reuse Lambda Stack's preinstalled PyTorch/CUDA packages
while keeping project-specific Python packages isolated in the venv.

Next time, activate it with:
  source $VENV_DIR/bin/activate
EOF
