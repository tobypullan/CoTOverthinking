# Running `baseline_CoT_options.py` On Lambda

This directory contains a local `transformers` version of the `baseline_CoT_options.py` experiment. On Lambda, you can run the same experiment directly on a Lambda GPU instance without the Bede-specific login nodes or Slurm workflow.

These instructions assume you launched the default Lambda Stack image on Lambda Cloud. On current Lambda on-demand instances, that image already includes Ubuntu, NVIDIA drivers, CUDA/cuDNN/NCCL, Docker, JupyterLab, PyTorch, TensorFlow, JAX, `tmux`, and other dev tools, so the setup flow should reuse that stack instead of reinstalling it from scratch.

These notes now also include an optional Ollama setup for `gemma3:4b`. That is useful for local benchmarking and fast warm inference on a Lambda GPU instance, but it is a separate runtime from the current Python script. The current `baseline_CoT_options.py` still runs through Hugging Face `transformers`.

This guide keeps the same:

- script: `CoT_Overthinking_Bede/baseline_CoT_options.py`
- model: `google/gemma-3-4b-it`
- dataset: `TIGER-Lab/MMLU-Pro`
- output format and default output path: `CoT_Overthinking_Bede/baseline/baseline_CoTs_options.jsonl`

## What Stays The Same

- The main experiment still runs locally through Hugging Face `transformers`.
- The default model in `baseline_CoT_options.py` is still `google/gemma-3-4b-it`.
- The prompt format is unchanged, including the explicit `<think>...</think>` reasoning instruction.
- The output JSONL is intentionally minimal: `question_id`, `category`, `question`, `prompt`, `response`, `parsed_answer`.

## What Changes On Lambda

- You do not need `ghlogin`, `ghtest`, `gh`, or `sbatch`.
- You run the script directly from your Lambda machine.
- On the default Lambda Stack image, the GPU driver and CUDA stack are already installed.
- `tmux`, `screen`, and Docker are already present on the default image, so there is no extra OS-level setup for those tools.
- If you want an alternative local runtime for fast warm inference, you can optionally install Ollama and pull `gemma3:4b`.

## 1. Get The Repo Onto The Lambda Machine

If the repo is not already there:

```bash
cd "$HOME"
git clone <your_repo_url> CoTOverthinking
cd CoTOverthinking
```

If it is already present, just `cd` into it.

Before you terminate a paid Lambda VM, commit and push any repo changes that you want to keep:

```bash
cd "$HOME/CoTOverthinking"
git status
git add CoT_Overthinking_Bede
git commit -m "Save Lambda setup and experiment updates"
git push
```

A pushed Git commit preserves code and docs, but it does not preserve:

- Hugging Face cache downloads
- Ollama model blobs
- virtual environments
- uncommitted outputs or logs

If you want those to survive instance termination, put them on a Lambda filesystem or another persistent disk.

## 2. Create The Python Environment

Use the Lambda-specific helper:

```bash
cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"
bash setup_lambda_env.sh
```

That creates:

- venv: `$HOME/.venvs/cot-overthinking-lambda`
- Hugging Face cache root: `$HOME/.cache/huggingface`

Unlike the Bede helper, this script creates the venv with `--system-site-packages` so it can reuse Lambda Stack's preinstalled PyTorch/CUDA packages. It then installs only the repo's Python requirements into the venv.

If you want a custom venv location:

```bash
bash setup_lambda_env.sh "$HOME/my-gemma-venv"
```

If you prefer to do the setup manually instead of using the helper script:

```bash
python3 -m venv --system-site-packages "$HOME/.venvs/cot-overthinking-lambda"
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$HOME/CoTOverthinking/CoT_Overthinking_Bede/requirements.txt"
```

If `import torch` fails after that, you are probably not on the default Lambda Stack image. In that case, switch to Lambda Stack or do a separate manual setup for the image you chose.

## 3. Set Up Hugging Face Cache Directories

```bash
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
```

If this Lambda instance will be reused, add those exports to your shell init file. If you attached a Lambda filesystem, consider pointing `HF_HOME` at `/lambda/nfs/<FILESYSTEM_NAME>/huggingface` so model downloads survive instance termination.

## 4. Accept The Gemma License And Log In To Hugging Face

`google/gemma-3-4b-it` is gated on Hugging Face, so do both of these before the first run:

1. Accept the Gemma terms on:

```text
https://huggingface.co/google/gemma-3-4b-it
```

2. Log in on the Lambda machine:

```bash
huggingface-cli login
```

## 5. Optional: Pre-Download The Model And Dataset

This is recommended so the first actual experiment run does not spend time downloading assets.

```bash
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"

python3 - <<'PY'
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_id = "google/gemma-3-4b-it"
AutoTokenizer.from_pretrained(model_id)
snapshot_download(repo_id=model_id)
load_dataset("TIGER-Lab/MMLU-Pro", split="test")
print("Cache warm.")
PY
```

## 6. Confirm That PyTorch Can See The GPU

Before running the experiment, check that CUDA is available:

```bash
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
python3 - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name =", torch.cuda.get_device_name(0))
PY
```

You can also inspect the GPU with:

```bash
nvidia-smi
```

## 7. Run A Small Smoke Test

From the Lambda machine:

```bash
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"

python3 baseline_CoT_options.py --question-ids 0 --model-id google/gemma-3-4b-it
```

If you want an even smaller smoke test first, you can temporarily use:

```bash
python3 baseline_CoT_options.py --question-ids 0 --model-id google/gemma-3-1b-it
```

If you want the more stable short-run timing mode that avoids Transformers' decode auto-compile behaviour, use:

```bash
python3 baseline_CoT_options.py --question-ids 0 --max-new-tokens 128 --disable-compile
```

## 8. Run The Full Experiment Directly

Example category slice:

```bash
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"

python3 baseline_CoT_options.py \
  --model-id google/gemma-3-4b-it \
  --category math \
  --start 0 \
  --end 10
```

For a longer run that keeps going after disconnect, use `tmux` or `screen`.

Example with `tmux`:

```bash
tmux new -s cot-gemma
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd "$HOME/CoTOverthinking/CoT_Overthinking_Bede"

python3 baseline_CoT_options.py \
  --model-id google/gemma-3-4b-it \
  --category math \
  --start 0 \
  --end 100
```

Detach with `Ctrl-b` then `d`.

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

Use the more stable short-run mode:

```bash
python3 baseline_CoT_options.py --category math --start 0 --end 100 --disable-compile
```

## 9. Optional: Set Up Ollama With `gemma3:4b`

Use this if you want a separate local inference runtime for benchmarking or serving. This is not yet wired into `baseline_CoT_options.py`, but it is useful to have available on a fresh Lambda VM.

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh -o /tmp/ollama-install.sh
sh /tmp/ollama-install.sh
```

Confirm that the service is up:

```bash
ollama --version
systemctl status ollama --no-pager -n 40
curl http://127.0.0.1:11434/api/version
```

Pull the model:

```bash
ollama pull gemma3:4b
ollama list
```

Run a quick CLI smoke test:

```bash
ollama run gemma3:4b "Reply with exactly OK"
```

Run a quick API smoke test and keep the model warm for five minutes:

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Reply with exactly OK",
  "stream": false,
  "keep_alive": "5m"
}'
```

Check whether the model is resident on the GPU:

```bash
ollama ps
nvidia-smi
```

Notes for Ollama on Lambda:

- The first cold runner startup can be slow. On one A10 test VM, the first model startup took about a minute before warm requests became fast.
- After the model is loaded, `ollama ps` should show `gemma3:4b` using `100% GPU`.
- The systemd service stores model blobs under `/usr/share/ollama/.ollama/models` by default.
- Those model blobs are not stored in Git. On a fresh VM, you will usually need to run `ollama pull gemma3:4b` again unless that directory lives on persistent storage.
- If you want Ollama models to survive instance termination, move or bind that model directory onto a persistent Lambda filesystem.

## 10. Optional: Reproduce The Ollama Benchmark

Warm the model once:

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Reply with exactly OK",
  "stream": false,
  "keep_alive": "10m"
}'
```

Then benchmark from Python:

```bash
source "$HOME/.venvs/cot-overthinking-lambda/bin/activate"
cd "$HOME/CoTOverthinking"

python3 - <<'PY'
import json
import urllib.request
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "baseline_cot_options",
    Path("CoT_Overthinking_Bede/baseline_CoT_options.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def call_ollama(prompt):
    payload = {
        "model": "gemma3:4b",
        "system": mod.SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": 128,
            "temperature": mod.FULL_TRACE_TEMPERATURE,
            "top_p": mod.FULL_TRACE_TOP_P,
            "top_k": mod.FULL_TRACE_TOP_K,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))

ds = mod.get_dataset()
total_tokens = 0
total_seconds = 0.0
for q_id in range(10):
    prompt = mod.build_options_prompt(ds[q_id]["question"], ds[q_id]["options"])
    out = call_ollama(prompt)
    seconds = out["total_duration"] / 1e9
    tokens = out["eval_count"]
    total_tokens += tokens
    total_seconds += seconds
    print(q_id, round(tokens / seconds, 2), "tok/s", "done_reason=", out["done_reason"])

print("weighted tok/s =", round(total_tokens / total_seconds, 2))
PY
```

## Notes For Lambda

- The experiment code itself does not need to change for Lambda.
- The main difference from Bede is operational: direct shell execution instead of partition selection and batch submission.
- These instructions are written for the default Lambda Stack image. If you launch `GPU Base` or plain `Ubuntu Server`, you will need a different environment setup.
- Lambda instance root storage is ephemeral. If you want model downloads and outputs to persist after termination, put `HF_HOME` and your output path on a Lambda filesystem such as `/lambda/nfs/<FILESYSTEM_NAME>`.
- Ollama models are also ephemeral unless you place `/usr/share/ollama/.ollama/models` or an alternative Ollama model directory on persistent storage.
- Avoid `sudo do-release-upgrade` or replacing the preinstalled system Python on Lambda, because that can break JupyterLab on the instance.

## Sources

- Lambda on-demand images overview: https://docs.lambda.ai/public-cloud/on-demand/
- Lambda virtual environment guidance: https://docs.lambda.ai/education/programming/virtual-environments-containers/
- Lambda troubleshooting on cuDNN / Lambda Stack Python packages: https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/
- Lambda getting started FAQ: https://docs.lambda.ai/public-cloud/on-demand/getting-started/
- Ollama install: https://ollama.com/download/linux
- Ollama API reference: https://docs.ollama.com/api
- Ollama model library entry for Gemma 3: https://ollama.com/library/gemma3
