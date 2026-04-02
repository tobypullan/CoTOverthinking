#!/bin/bash
#SBATCH --account=bdXXXXX
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=cot-options
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CoTOverthinking}"
WORKDIR="${REPO_ROOT}/CoT_Overthinking_Bede"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/cot-overthinking-bede}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

source "${VENV_DIR}/bin/activate"
cd "${WORKDIR}"

python3 baseline_CoT_options.py \
  --category math \
  --start 0 \
  --end 10
