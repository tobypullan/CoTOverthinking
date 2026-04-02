# Running `baseline_CoT_options.py` On Bede

This directory contains a Bede-native replacement for the Groq-backed `baseline_CoT_options.py` experiment. It runs `Qwen/Qwen3-32B` locally with Hugging Face `transformers` on a Grace Hopper GPU.

## What Changed

- The Groq client has been removed.
- Inference now runs locally through `transformers` on a Bede GPU node.
- The output schema is kept close to the original JSONL so it is easy to compare with your existing results.
- The script writes to `CoT_Overthinking_Bede/baseline/baseline_CoTs_options.jsonl` by default.

## Recommended Bede Target

Use the Grace Hopper environment:

- `ghlogin` for setup work
- `ghtest` for short GPU smoke tests
- `gh` for full runs

This matches Bede’s current architecture guidance:

- `ghlogin` is CPU-only as of **2025-09-23**
- `gh` / `ghtest` are the `aarch64` GPU partitions
- `--gres=gpu:1` on `gh` gives one GH200 GPU, plus the full node CPU and memory allocation

## 1. Copy The Repo To Your Home Directory

You said you do not have access to the Bede project directory, so this setup assumes everything lives under your home directory.

Example:

```bash
cd "$HOME"
git clone <your_repo_url> CoTOverthinking
cd CoTOverthinking
```

If the repo is already on Bede, skip this.

## 2. Start A Grace-Hopper Login Session

From the normal Bede login node:

```bash
ghlogin -A <project>
```

For more memory / CPU on the login session:

```bash
ghlogin -A <project> --time 4:00:00 -c 8 --mem 24G
```

Use `ghlogin` for environment setup and downloads only. It does not give you a GPU.

## 3. Create The Python Environment

From inside `ghlogin`:

```bash
cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"
bash setup_bede_env.sh
```

That script will:

- create a venv in `$HOME/.venvs/cot-overthinking-bede`
- install CUDA-enabled PyTorch from the Bede-compatible `cu124` wheel index
- install the rest of the Python requirements from `requirements.txt`

If you want a custom venv path:

```bash
bash setup_bede_env.sh "$HOME/my-qwen-venv"
```

## 4. Set Up Hugging Face Cache Directories

Use your home directory for Hugging Face cache data:

```bash
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
```

If you use these often, add them to your shell init on Bede.

## 5. Optional: Pre-Download The Model And Dataset

You can let the batch job download everything the first time, but it is cleaner to warm the cache from `ghlogin` first:

```bash
source "$HOME/.venvs/cot-overthinking-bede/bin/activate"
cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"

python3 - <<'PY'
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-32B"
AutoTokenizer.from_pretrained(model_id)
snapshot_download(repo_id=model_id)
load_dataset("TIGER-Lab/MMLU-Pro", split="test")
print("Cache warm.")
PY
```

This step is optional, but it avoids mixing long model downloads into your first batch job.

## 6. Smoke Test On `ghtest`

Use a short GPU test before a long run.

Inside `ghlogin`:

```bash
srun -A <project> -p ghtest --gres=gpu:1 --time=00:20:00 --pty bash
```

Then inside that GPU shell:

```bash
source "$HOME/.venvs/cot-overthinking-bede/bin/activate"
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"

python3 baseline_CoT_options.py --question-ids 0 --model-id Qwen/Qwen3-32B
```

If you want a lighter smoke test, swap in `--model-id Qwen/Qwen3-8B`.

## 7. Submit A Full Batch Job

Edit [`run_baseline_CoT_options.slurm`](./run_baseline_CoT_options.slurm):

- replace `bdXXXXX` with your actual project code
- if needed, adjust `REPO_ROOT`
- change the Python arguments at the bottom to match the slice you want

Example submission from inside `ghlogin`:

```bash
cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"
sbatch run_baseline_CoT_options.slurm
```

Or from a normal Bede login node:

```bash
cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"
ghbatch run_baseline_CoT_options.slurm
```

## Example Slurm Script

The provided file is:

```bash
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
```

## Useful Run Variants

Run a category slice:

```bash
python3 baseline_CoT_options.py --category math --start 0 --end 100
```

Run specific question ids:

```bash
python3 baseline_CoT_options.py --question-ids 0 1 2 3
```

Resume an interrupted run without regenerating finished rows:

```bash
python3 baseline_CoT_options.py --category math --start 0 --end 100
```

Force regeneration:

```bash
python3 baseline_CoT_options.py --category math --start 0 --end 100 --overwrite
```

Use only already-cached model files:

```bash
python3 baseline_CoT_options.py --category math --local-files-only
```

## Output Location

Default output:

```text
CoT_Overthinking_Bede/baseline/baseline_CoTs_options.jsonl
```

Override it if needed:

```bash
python3 baseline_CoT_options.py \
  --category math \
  --output-path "$HOME/my_outputs/baseline_CoTs_options.jsonl"
```

## Notes And Caveats

- This version uses local Hugging Face generation, not Groq. Expect some answer traces to differ from your earlier Groq outputs even with the same base model family.
- The script uses Qwen3 thinking mode for the main pass and non-thinking mode for the label-extraction judge fallback.
- If you accidentally run it on `ghlogin` without a GPU allocation, it will fail fast with a CUDA error message.
- If you hit memory pressure, reduce the job size first by testing with fewer questions or a smaller model such as `Qwen/Qwen3-8B`.
- Because this setup now uses your home directory for the venv and Hugging Face caches, make sure your home quota is large enough for the model and dataset files.
- If you want the longest possible GH200 memory headroom, Bede also supports explicitly requesting `--gres=gpu:gh200_144g:1`, but that will usually queue longer than a generic `gpu:1`.

## Sources

- Bede usage guide: https://bede-documentation.readthedocs.io/en/latest/usage/index.html
- Bede Conda guide: https://bede-documentation.readthedocs.io/en/latest/software/applications/conda.html
- Bede PyTorch guide: https://bede-documentation.readthedocs.io/en/latest/software/applications/pytorch.html
- Qwen quickstart: https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html
