import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


OLLAMA_MODEL_ID = "gemma3:4b"
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
BASELINE_OPTIONS_PATH = SCRIPT_DIR / "baseline" / "baseline_CoTs_options_ollama.jsonl"
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


class OllamaRunner:
    def __init__(
        self,
        model_id,
        *,
        base_url="http://127.0.0.1:11434",
        timeout_seconds=600,
        keep_alive="10m",
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def generate(
        self,
        messages,
        *,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
    ):
        prompt_prep_start = time.perf_counter()
        system_text = ""
        user_text_parts = []

        for message in messages:
            role = message.get("role")
            content = message.get("content") or []
            text = "".join(
                part.get("text", "")
                for part in content
                if part.get("type") == "text"
            )
            if role == "system":
                system_text = text
            elif role == "user":
                user_text_parts.append(text)

        prompt_prep_seconds = time.perf_counter() - prompt_prep_start
        payload = {
            "model": self.model_id,
            "system": system_text,
            "prompt": "\n\n".join(part for part in user_text_parts if part),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        generation_start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Failed to reach the Ollama API. Ensure the Ollama service is running "
                f"and the model {self.model_id!r} is available."
            ) from exc
        generation_seconds = time.perf_counter() - generation_start

        eval_count = raw_result.get("eval_count")
        eval_duration_ns = raw_result.get("eval_duration")
        prompt_eval_count = raw_result.get("prompt_eval_count")
        prompt_eval_duration_ns = raw_result.get("prompt_eval_duration")

        tokens_per_second = None
        if eval_count and eval_duration_ns:
            eval_duration_seconds = eval_duration_ns / 1_000_000_000
            if eval_duration_seconds > 0:
                tokens_per_second = eval_count / eval_duration_seconds
        elif eval_count and generation_seconds > 0:
            tokens_per_second = eval_count / generation_seconds

        return {
            "raw_response": (raw_result.get("response") or "").strip(),
            "finish_reason": raw_result.get("done_reason", "stop"),
            "generated_token_count": eval_count if eval_count is not None else 0,
            "effective_max_new_tokens": max_new_tokens,
            "prompt_token_count": (
                prompt_eval_count
                if prompt_eval_count is not None
                else 0
            ),
            "prompt_prep_seconds": prompt_prep_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second": tokens_per_second,
            "prompt_eval_seconds": (
                prompt_eval_duration_ns / 1_000_000_000
                if prompt_eval_duration_ns is not None
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
    actual_answer_label = ds[q_id]["answer"]
    prompt = build_options_prompt(question, options)

    completion = runner.generate(
        messages=build_chat_messages(SYSTEM_PROMPT, prompt),
        temperature=FULL_TRACE_TEMPERATURE,
        top_p=FULL_TRACE_TOP_P,
        top_k=FULL_TRACE_TOP_K,
        max_new_tokens=args.max_new_tokens,
    )

    raw_response = completion["raw_response"]
    parsed_answer = parse_answer_from_response(raw_response)
    correct = parsed_answer == actual_answer_label

    return (
        {
            "question_id": q_id,
            "category": category,
            "question": question,
            "prompt": prompt,
            "response": raw_response,
            "parsed_answer": parsed_answer,
            "actual_answer_label": actual_answer_label,
            "correct": correct,
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


def build_runner(args):
    return OllamaRunner(
        model_id=args.model_id,
        base_url=args.ollama_base_url,
        timeout_seconds=args.ollama_timeout_seconds,
        keep_alive=args.ollama_keep_alive,
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
        "actual_answer_label",
        "correct",
    }
    return set(row) == required_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline Gemma 3 4B CoTs for MMLU-Pro options prompts with Ollama."
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
        default=OLLAMA_MODEL_ID,
        help=f"Ollama model identifier. Default: {OLLAMA_MODEL_ID!r}.",
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
        help="Maximum generated tokens for the main CoT pass.",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Base URL for the local Ollama API.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=int,
        default=600,
        help="HTTP timeout in seconds for Ollama generation requests.",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        type=str,
        default="10m",
        help="Ollama keep_alive value to keep the model warm between requests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ds = get_dataset()
    run_start = time.perf_counter()

    if not get_category_qids(args.category):
        raise RuntimeError(
            f"No questions found for category={args.category!r}. "
            "Use 'all' to disable category filtering."
        )

    print(f"Loading Ollama model={args.model_id}...")
    runner = build_runner(args)
    print("Runner ready.")

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
    processed = 0
    total_generated_tokens = 0
    total_generation_seconds = 0.0

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
        processed += 1
        total_generated_tokens += perf_stats["generated_token_count"]
        total_generation_seconds += perf_stats["generation_seconds"]
        tok_s = perf_stats["tokens_per_second"]
        tok_s_text = f"{tok_s:.2f}" if tok_s is not None else "n/a"
        print(
            f"Saved question {q_id}. "
            f"parsed_answer={baseline_row['parsed_answer']} "
            f"actual_answer_label={baseline_row['actual_answer_label']} "
            f"correct={baseline_row['correct']} "
            f"generated_tokens={perf_stats['generated_token_count']} "
            f"finish_reason={perf_stats['finish_reason']} "
            f"tok/s={tok_s_text} "
            f"prompt_prep_ms={perf_stats['prompt_prep_seconds'] * 1000:.1f}"
        )

    wall_seconds = time.perf_counter() - run_start
    weighted_toks = (
        total_generated_tokens / total_generation_seconds
        if total_generation_seconds > 0
        else None
    )
    weighted_toks_text = f"{weighted_toks:.2f}" if weighted_toks is not None else "n/a"
    print(
        "Run summary: "
        f"backend=ollama "
        f"model={args.model_id} "
        f"processed={processed} "
        f"output_path={args.output_path} "
        f"wall_seconds={wall_seconds:.2f} "
        f"generation_seconds={total_generation_seconds:.2f} "
        f"generated_tokens={total_generated_tokens} "
        f"weighted_tok_s={weighted_toks_text}"
    )


if __name__ == "__main__":
    main()
