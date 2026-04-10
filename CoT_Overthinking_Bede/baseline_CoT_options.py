import argparse
import json
import re
import time
from pathlib import Path


MODEL_ID = "google/gemma-3-4b-it"
OPTIONS_DEFAULT_DATASET_CATEGORY = "math"

SYSTEM_PROMPT = (
    "Solve the following multiple-choice problem. First write your reasoning "
    "inside <think> and </think>. After the closing </think> tag, output only "
    "a single uppercase letter corresponding to the correct answer choice."
)

FULL_TRACE_TEMPERATURE = 0.6
FULL_TRACE_TOP_P = 0.95
FULL_TRACE_TOP_K = 20
FULL_TRACE_MAX_COMPLETION_TOKENS = 8192

LABELS = tuple("ABCDEFGHIJ")
LABEL_SET = set(LABELS)

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_OPTIONS_PATH = SCRIPT_DIR / "baseline" / "baseline_CoTs_options.jsonl"
DATASET_NAME = "TIGER-Lab/MMLU-Pro"
DATASET_SPLIT = "test"

_DATASET = None


def normalise_category(category):
    return (category or "").strip().casefold()


def category_value_matches(row_category, requested_category):
    if normalise_category(requested_category) == "all":
        return True
    return normalise_category(row_category) == normalise_category(requested_category)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_num}: {exc}") from exc


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def build_options_prompt(question, options):
    labelled_options = [f"\n{LABELS[i]}: {options[i]}" for i in range(len(options))]
    return question + " The options are: " + "".join(labelled_options)


def extract_answer_label(text):
    if not text:
        return None

    text = text.strip()
    patterns = [
        r"^\**\s*\[?([A-J])\]?\s*\**$",
        r"\bfinal answer\s*(?:is|:)\s*\*{0,2}\[?([A-J])\]?\*{0,2}\b",
        r"ANSWER:\s*\[?([A-J])\]?\b",
        r"\banswer\s*(?:is|:)\s*\*{0,2}\[?([A-J])\]?\*{0,2}\b",
        r"\boption\s+([A-J])\b",
        r"\b([A-J])\b[\s\]\).,:;!?]*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    stripped = text.upper()
    if stripped in LABEL_SET:
        return stripped

    return None


def extract_post_think_text(response):
    if not response:
        return ""

    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()

    return response.strip()


def parse_answer_from_response(response):
    if not response:
        return "UNKNOWN"

    search_spaces = []

    post_think_text = extract_post_think_text(response)
    if post_think_text:
        search_spaces.append(post_think_text)

    stripped_response = response.strip()
    if stripped_response:
        last_line = stripped_response.splitlines()[-1].strip()
        if last_line and last_line not in search_spaces:
            search_spaces.append(last_line)

    for search_text in search_spaces:
        label = extract_answer_label(search_text)
        if label:
            return label

    return "UNKNOWN"


def build_chat_messages(system_text, user_text):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


class LocalGemmaRunner:
    def __init__(self, model_id, local_files_only=False, torch_dtype="auto"):
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. On Bede, `ghlogin` is CPU-only as of "
                "2025-09-23. Run this inside a `gh`/`ghtest` GPU allocation."
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.torch = torch
        self.model_id = model_id
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        )
        self.model.eval()
        text_config = getattr(self.model.config, "text_config", None)
        self.max_context_tokens = getattr(
            text_config,
            "max_position_embeddings",
            getattr(self.model.config, "max_position_embeddings", None),
        )

    def _resolve_torch_dtype(self, torch_dtype):
        if torch_dtype == "auto":
            return "auto"

        dtype_map = {
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        try:
            return dtype_map[torch_dtype.casefold()]
        except KeyError as exc:
            supported = ", ".join(sorted(dtype_map))
            raise ValueError(
                f"Unsupported --torch-dtype value {torch_dtype!r}. "
                f"Expected one of: auto, {supported}."
            ) from exc

    def _normalise_generated_text(self, generated_text):
        generated_text = generated_text.strip("\n")
        if "<think>" in generated_text:
            return generated_text.strip()
        return generated_text.strip()

    def generate(
        self,
        messages,
        *,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        disable_compile=False,
    ):
        prompt_prep_start = time.perf_counter()
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        prompt_prep_seconds = time.perf_counter() - prompt_prep_start
        input_length = model_inputs["input_ids"].shape[-1]

        effective_max_new_tokens = max_new_tokens
        if self.max_context_tokens is not None:
            remaining = self.max_context_tokens - input_length
            if remaining <= 0:
                raise RuntimeError(
                    "Prompt length already exceeds the model context window. "
                    "Try a shorter prompt or a model with a larger context."
                )
            effective_max_new_tokens = min(effective_max_new_tokens, remaining)

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": effective_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k
        if disable_compile:
            generation_kwargs["disable_compile"] = True

        self.torch.cuda.synchronize()
        generation_start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**model_inputs, **generation_kwargs)
        self.torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_start

        sequences = outputs.sequences[0]
        output_ids = sequences[input_length:].tolist()
        generated_text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_response = self._normalise_generated_text(generated_text)

        finish_reason = "length" if len(output_ids) >= effective_max_new_tokens else "stop"
        return {
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "generated_token_count": len(output_ids),
            "effective_max_new_tokens": effective_max_new_tokens,
            "prompt_token_count": input_length,
            "prompt_prep_seconds": prompt_prep_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second": (
                len(output_ids) / generation_seconds
                if generation_seconds > 0
                else None
            ),
        }

def get_dataset():
    global _DATASET
    if _DATASET is None:
        from datasets import load_dataset

        _DATASET = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return _DATASET


def get_row_category(q_id):
    ds = get_dataset()
    return ds[q_id].get("category")


def category_matches(q_id, category):
    return category_value_matches(get_row_category(q_id), category)


def get_category_qids(category):
    ds = get_dataset()
    return [q_id for q_id in range(len(ds)) if category_matches(q_id, category)]


def baseline_cot(q_id, runner, args):
    ds = get_dataset()
    question = ds[q_id]["question"]
    options = ds[q_id]["options"]
    category = get_row_category(q_id)
    prompt = build_options_prompt(question, options)

    completion = runner.generate(
        messages=build_chat_messages(SYSTEM_PROMPT, prompt),
        temperature=FULL_TRACE_TEMPERATURE,
        top_p=FULL_TRACE_TOP_P,
        top_k=FULL_TRACE_TOP_K,
        max_new_tokens=args.max_new_tokens,
        disable_compile=args.disable_compile,
    )

    raw_response = completion["raw_response"]
    parsed_answer = parse_answer_from_response(raw_response)

    return (
        {
            "question_id": q_id,
            "category": category,
            "question": question,
            "prompt": prompt,
            "response": raw_response,
            "parsed_answer": parsed_answer,
        },
        {
            "finish_reason": completion["finish_reason"],
            "prompt_token_count": completion["prompt_token_count"],
            "generated_token_count": completion["generated_token_count"],
            "prompt_prep_seconds": completion["prompt_prep_seconds"],
            "generation_seconds": completion["generation_seconds"],
            "tokens_per_second": completion["tokens_per_second"],
        },
    )


def iter_requested_qids(args):
    ds = get_dataset()
    if args.question_ids:
        requested = []
        for q_id in args.question_ids:
            if q_id < 0 or q_id >= len(ds):
                requested.append(q_id)
                continue
            requested.append(q_id)
        return requested
    category_qids = get_category_qids(args.category)
    return category_qids[args.start : args.end]


def load_existing_baseline_rows(output_path):
    if not output_path.exists():
        return {}
    return {
        row["question_id"]: row
        for row in read_jsonl(output_path)
        if "question_id" in row
    }


def row_is_compatible(row):
    required_keys = {
        "question_id",
        "category",
        "question",
        "prompt",
        "response",
        "parsed_answer",
    }
    return set(row) == required_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline Gemma 3 4B CoTs for MMLU-Pro options prompts on Bede."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inclusive start index within the selected category subset.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end index within the selected category subset. Default: end of slice.",
    )
    parser.add_argument(
        "--question-ids",
        type=int,
        nargs="+",
        help="Specific raw dataset question ids to run instead of a category slice.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=OPTIONS_DEFAULT_DATASET_CATEGORY,
        help=(
            "Dataset category to run. Use 'all' to disable category filtering. "
            f"Default: {OPTIONS_DEFAULT_DATASET_CATEGORY}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate rows even if they already exist in the baseline file.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"Hugging Face model id to run. Default: {MODEL_ID}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=BASELINE_OPTIONS_PATH,
        help=f"Output JSONL path. Default: {BASELINE_OPTIONS_PATH}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=FULL_TRACE_MAX_COMPLETION_TOKENS,
        help=(
            "Maximum generated tokens for the main CoT pass before context-window "
            "capping is applied."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model/tokenizer from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype passed to from_pretrained, e.g. auto or bfloat16.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help=(
            "Disable Transformers' automatic decode compilation. Useful for short "
            "smoke tests where cold-start latency matters more than peak "
            "steady-state tokens/second."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ds = get_dataset()

    if not get_category_qids(args.category):
        raise RuntimeError(
            f"No questions found for category={args.category!r}. "
            "Use 'all' to disable category filtering."
        )

    print(f"Loading model {args.model_id}...")
    runner = LocalGemmaRunner(
        model_id=args.model_id,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
    )
    print("Model ready.")

    existing_rows = load_existing_baseline_rows(args.output_path)
    existing_ids = (
        set()
        if args.overwrite
        else {
            q_id
            for q_id, row in existing_rows.items()
            if row_is_compatible(row)
        }
    )

    for q_id in iter_requested_qids(args):
        if q_id < 0 or q_id >= len(ds):
            print(f"Skipping out-of-range question {q_id}.")
            continue
        if q_id in existing_ids:
            print(f"Skipping question {q_id}; baseline already exists.")
            continue

        print(f"Processing question {q_id}...")
        baseline_row, perf_stats = baseline_cot(q_id, runner, args)
        existing_rows[q_id] = baseline_row
        write_jsonl(
            args.output_path,
            [existing_rows[key] for key in sorted(existing_rows)],
        )
        existing_ids.add(q_id)
        print(
            f"Saved question {q_id}. "
            f"parsed_answer={baseline_row['parsed_answer']} "
            f"generated_tokens={perf_stats['generated_token_count']} "
            f"finish_reason={perf_stats['finish_reason']} "
            f"tok/s={perf_stats['tokens_per_second']:.2f} "
            f"prompt_prep_ms={perf_stats['prompt_prep_seconds'] * 1000:.1f}"
        )


if __name__ == "__main__":
    main()
