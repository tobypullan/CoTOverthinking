import argparse
import json
import re
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-32B"
OPTIONS_BASELINE_SCHEMA_VERSION = 1
OPTIONS_DEFAULT_DATASET_CATEGORY = "math"

SYSTEM_PROMPT = (
    "Solve the following problem. Please make sure that your response only "
    "consists of a single letter corresponding to the correct answer choice. "
    "Do not include anything else in your final response."
)

FULL_TRACE_TEMPERATURE = 0.6
FULL_TRACE_TOP_P = 0.95
FULL_TRACE_TOP_K = 20
FULL_TRACE_MAX_COMPLETION_TOKENS = 38912

JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_TOP_K = 20
JUDGE_MAX_COMPLETION_TOKENS = 8

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


def extract_reasoning_trace(response):
    if not response:
        return None

    closed_match = re.search(r"<think>\s*(.*?)\s*</think>", response, flags=re.DOTALL)
    if closed_match:
        return closed_match.group(1).strip()

    start = response.find("<think>")
    if start == -1:
        return None

    return response[start + len("<think>") :].strip()


def extract_final_answer_text(response):
    if not response:
        return ""

    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()

    return response.strip()


class LocalQwenRunner:
    def __init__(self, model_id, local_files_only=False, torch_dtype="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. On Bede, `ghlogin` is CPU-only as of "
                "2025-09-23. Run this inside a `gh`/`ghtest` GPU allocation."
            )

        self.torch = torch
        self.model_id = model_id
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.max_context_tokens = getattr(self.model.config, "max_position_embeddings", None)

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

    def _build_prompt(self, messages, enable_thinking):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError as exc:
            raise RuntimeError(
                "This script requires `transformers>=4.51.0` because it relies "
                "on Qwen3 chat templates with `enable_thinking=`."
            ) from exc

    def _normalise_generated_text(self, generated_text, enable_thinking):
        generated_text = generated_text.strip("\n")
        if not enable_thinking:
            return generated_text.strip()
        if "<think>" in generated_text:
            return generated_text
        return f"<think>\n{generated_text}".strip()

    def generate(
        self,
        messages,
        *,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        enable_thinking,
    ):
        prompt = self._build_prompt(messages, enable_thinking=enable_thinking)
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        model_inputs = {
            key: value.to(self.model.device)
            for key, value in model_inputs.items()
        }
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

        with self.torch.inference_mode():
            outputs = self.model.generate(**model_inputs, **generation_kwargs)

        sequences = outputs.sequences[0]
        output_ids = sequences[input_length:].tolist()
        generated_text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_response = self._normalise_generated_text(
            generated_text,
            enable_thinking=enable_thinking,
        )

        finish_reason = "length" if len(output_ids) >= effective_max_new_tokens else "stop"
        return {
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "generated_token_count": len(output_ids),
            "effective_max_new_tokens": effective_max_new_tokens,
        }

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))


def resolve_answer_label(runner, response, options):
    direct_label = extract_answer_label(response)
    if direct_label:
        return {
            "judge_response": None,
            "answer": direct_label,
            "direct_parsed_answer": direct_label,
            "judge_parsed_answer": None,
            "used_llm_judge": False,
            "answer_source": "direct_parse",
        }

    labelled_options = [f"{LABELS[i]}: {options[i]}" for i in range(len(options))]
    judge_completion = runner.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract the answer label already chosen in another "
                    "model's response. Do not solve the question yourself. You "
                    "may use the provided options only to map quoted or "
                    "paraphrased option text back to a label. If the response "
                    "does not clearly choose exactly one option, return only "
                    "UNKNOWN."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Options: {labelled_options}\n\n"
                    f"Model response: {response}\n\n"
                    "Return only one token: A, B, C, D, E, F, G, H, I, J, or "
                    "UNKNOWN."
                ),
            },
        ],
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        top_k=JUDGE_TOP_K,
        max_new_tokens=JUDGE_MAX_COMPLETION_TOKENS,
        enable_thinking=False,
    )
    judge_response = judge_completion["raw_response"].strip()
    judged_label = extract_answer_label(judge_response)
    if judged_label:
        return {
            "judge_response": judge_response,
            "answer": judged_label,
            "direct_parsed_answer": "UNKNOWN",
            "judge_parsed_answer": judged_label,
            "used_llm_judge": True,
            "answer_source": "llm_judge",
        }

    return {
        "judge_response": judge_response,
        "answer": "UNKNOWN",
        "direct_parsed_answer": "UNKNOWN",
        "judge_parsed_answer": "UNKNOWN",
        "used_llm_judge": True,
        "answer_source": "unknown",
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
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=FULL_TRACE_TEMPERATURE,
        top_p=FULL_TRACE_TOP_P,
        top_k=FULL_TRACE_TOP_K,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=True,
    )

    raw_response = completion["raw_response"]
    finish_reason = completion["finish_reason"]
    reasoning_trace = extract_reasoning_trace(raw_response)
    reasoning_token_count = (
        runner.count_tokens(reasoning_trace)
        if reasoning_trace is not None
        else None
    )

    final_answer_text = ""
    judge_response = None
    direct_parsed_answer = "UNKNOWN"
    judge_parsed_answer = None
    used_llm_judge = False
    answer_source = "unknown"
    answer = "UNKNOWN"
    if finish_reason == "stop":
        final_answer_text = extract_final_answer_text(raw_response)
        resolution = resolve_answer_label(runner, final_answer_text, options)
        judge_response = resolution["judge_response"]
        direct_parsed_answer = resolution["direct_parsed_answer"]
        judge_parsed_answer = resolution["judge_parsed_answer"]
        used_llm_judge = resolution["used_llm_judge"]
        answer_source = resolution["answer_source"]
        answer = resolution["answer"]

    correct = answer == ds[q_id]["answer"]
    return {
        "question_id": q_id,
        "category": category,
        "question": question,
        "prompt": prompt,
        "response": raw_response,
        "reasoning_trace": reasoning_trace,
        "reasoning_token_count": reasoning_token_count,
        "final_answer_text": final_answer_text,
        "parsed_answer": f"ANSWER: {answer}" if answer != "UNKNOWN" else "UNKNOWN",
        "judge_response": judge_response,
        "direct_parsed_answer": direct_parsed_answer,
        "judge_parsed_answer": judge_parsed_answer,
        "used_llm_judge": used_llm_judge,
        "answer_source": answer_source,
        "answer": answer,
        "correct": correct,
        "actual_answer": ds[q_id]["answer"],
        "baseline_schema_version": OPTIONS_BASELINE_SCHEMA_VERSION,
        "complete_reason": finish_reason,
        "model_id": args.model_id,
        "generation_backend": "huggingface_transformers",
    }


def iter_requested_qids(args):
    ds = get_dataset()
    if args.question_ids:
        requested = []
        for q_id in args.question_ids:
            if q_id < 0 or q_id >= len(ds):
                requested.append(q_id)
                continue
            if not category_matches(q_id, args.category):
                print(
                    f"Skipping question {q_id}; category={get_row_category(q_id)!r} "
                    f"does not match requested category={args.category!r}."
                )
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


def row_is_compatible(row, expected_model_id):
    return (
        row.get("baseline_schema_version") == OPTIONS_BASELINE_SCHEMA_VERSION
        and row.get("category") is not None
        and "reasoning_token_count" in row
        and "final_answer_text" in row
        and "answer" in row
        and row.get("model_id", expected_model_id) == expected_model_id
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline Qwen3-32B CoTs for MMLU-Pro options prompts on Bede."
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
    runner = LocalQwenRunner(
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
            if row_is_compatible(row, args.model_id)
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
        baseline_row = baseline_cot(q_id, runner, args)
        existing_rows[q_id] = baseline_row
        write_jsonl(
            args.output_path,
            [existing_rows[key] for key in sorted(existing_rows)],
        )
        existing_ids.add(q_id)
        print(
            f"Saved question {q_id}. finish_reason={baseline_row['complete_reason']} "
            f"answer={baseline_row['answer']} correct={baseline_row['correct']}"
        )


if __name__ == "__main__":
    main()
