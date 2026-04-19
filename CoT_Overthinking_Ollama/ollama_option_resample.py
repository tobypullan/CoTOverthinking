import argparse
import hashlib
import json
import math
import random
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import (
    OPTIONS_DEFAULT_DATASET_CATEGORY,
    RANDOM_CONTROL_DEFAULT_SEED,
    SUPPORTED_RESAMPLE_CONDITIONS,
    category_value_matches,
    extract_answer_label,
    extract_reasoning_trace,
    read_jsonl,
    validate_resample_condition,
    write_jsonl,
)
from ollama_options import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT


OLLAMA_MODEL_ID = "gemma3:4b"
TOKENIZER_MODEL_ID = "google/gemma-3-4b-it"
OLLAMA_OPTIONS_PROBE_METHOD_VERSION = 3
OLLAMA_OPTIONS_RESAMPLE_SCHEMA_VERSION = 1
PROBE_TEMPERATURE = 0.0
PROBE_TOP_P = 1.0
PROBE_MAX_COMPLETION_TOKENS = 32
RESAMPLE_POINTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

BASELINE_OPTIONS_PATH = SCRIPT_DIR / "baseline" / "baseline_CoTs_options_ollama.jsonl"
OPTIONS_RESULTS_PATH = SCRIPT_DIR / "resample_results" / "options_results_ollama.jsonl"
RANDOM_OPTIONS_RESULTS_PATH = SCRIPT_DIR / "resample_results" / "options_random_results_ollama.jsonl"
SHUFFLE_OPTIONS_RESULTS_PATH = SCRIPT_DIR / "resample_results" / "options_shuffle_results_ollama.jsonl"

DATASET_NAME = "TIGER-Lab/MMLU-Pro"
DATASET_SPLIT = "test"


ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def normalise_baseline_path(path):
    return str(Path(path).expanduser().resolve())


def get_options_results_path(condition):
    condition = validate_resample_condition(condition)
    if condition == "original":
        return OPTIONS_RESULTS_PATH
    if condition == "random":
        return RANDOM_OPTIONS_RESULTS_PATH
    if condition == "shuffle":
        return SHUFFLE_OPTIONS_RESULTS_PATH
    raise AssertionError(f"Unhandled condition: {condition}")


@lru_cache(maxsize=None)
def get_tokenizer(tokenizer_model_id):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Token-based deciles require transformers>=4.51.0. Install "
            "transformers before running the Ollama options resample experiment."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_id,
            use_fast=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer {tokenizer_model_id!r}."
        ) from exc

    return tokenizer


def get_reasoning_token_ids(reasoning_trace, tokenizer_model_id):
    tokenizer = get_tokenizer(tokenizer_model_id)
    return tokenizer.encode(reasoning_trace or "", add_special_tokens=False)


def decode_token_ids(token_ids, tokenizer_model_id):
    tokenizer = get_tokenizer(tokenizer_model_id)
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def decile_prefix_token_count(total_tokens, point):
    if point <= 0:
        return 0
    if point >= 1:
        return total_tokens
    return math.ceil(total_tokens * point)


@lru_cache(maxsize=None)
def get_non_special_vocab_token_ids(tokenizer_model_id):
    tokenizer = get_tokenizer(tokenizer_model_id)
    special_ids = {
        int(token_id)
        for token_id in tokenizer.all_special_ids
        if token_id is not None
    }
    vocab_token_ids = sorted({int(token_id) for token_id in tokenizer.get_vocab().values()})
    return tuple(token_id for token_id in vocab_token_ids if token_id not in special_ids)


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_resample_condition_full_token_ids(
    condition,
    reasoning_token_ids,
    question_id,
    tokenizer_model_id,
    base_seed=RANDOM_CONTROL_DEFAULT_SEED,
):
    condition = validate_resample_condition(condition)
    source_token_ids = list(reasoning_token_ids or [])

    if condition == "original":
        return source_token_ids, {
            "source": "original_reasoning_trace",
            "tokenizer_model_id": tokenizer_model_id,
        }

    if condition == "random":
        population = get_non_special_vocab_token_ids(tokenizer_model_id)
        sequence_seed = stable_seed(
            "ollama_options_random_control",
            base_seed,
            question_id,
            len(source_token_ids),
            tokenizer_model_id,
        )
        rng = random.Random(sequence_seed)
        random_token_ids = [
            population[rng.randrange(len(population))]
            for _ in range(len(source_token_ids))
        ]
        return random_token_ids, {
            "source": "uniform_random_vocab_excluding_special",
            "base_seed": base_seed,
            "sequence_seed": sequence_seed,
            "tokenizer_model_id": tokenizer_model_id,
        }

    if condition == "shuffle":
        sequence_seed = stable_seed(
            "ollama_options_shuffle_control",
            base_seed,
            question_id,
            len(source_token_ids),
            tokenizer_model_id,
        )
        shuffled_token_ids = list(source_token_ids)
        rng = random.Random(sequence_seed)
        rng.shuffle(shuffled_token_ids)
        return shuffled_token_ids, {
            "source": "original_reasoning_trace_token_permutation",
            "base_seed": base_seed,
            "sequence_seed": sequence_seed,
            "tokenizer_model_id": tokenizer_model_id,
        }

    raise AssertionError(f"Unhandled condition: {condition}")


class OllamaChatClient:
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

    def chat(
        self,
        messages,
        *,
        temperature,
        top_p,
        max_new_tokens,
    ):
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Failed to reach the Ollama API. Ensure the Ollama service is running "
                f"and the model {self.model_id!r} is available."
            ) from exc

        message = raw_result.get("message") or {}
        return (message.get("content") or "").strip()


def resolve_answer_without_judge(response):
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

    return {
        "judge_response": None,
        "answer": "UNKNOWN",
        "direct_parsed_answer": "UNKNOWN",
        "judge_parsed_answer": None,
        "used_llm_judge": False,
        "answer_source": "unknown",
    }


def build_forced_continuation_messages(prompt, reasoning_prefix):
    prefix_text = (reasoning_prefix or "").strip()
    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>{prefix_text}</think>\n"},
    ]


def probe_answer(client, prompt, reasoning_prefix):
    messages = build_forced_continuation_messages(prompt, reasoning_prefix)
    raw_probe_response = client.chat(
        messages,
        temperature=PROBE_TEMPERATURE,
        top_p=PROBE_TOP_P,
        max_new_tokens=PROBE_MAX_COMPLETION_TOKENS,
    )
    return raw_probe_response, raw_probe_response.strip()


def get_row_category(q_id):
    return ds[q_id].get("category")


def category_matches(q_id, category):
    return category_value_matches(get_row_category(q_id), category)


def get_category_qids(category):
    return [q_id for q_id in range(len(ds)) if category_matches(q_id, category)]


def load_baselines(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Baseline file not found at {path}. Run the Ollama baseline script first "
            "or pass --baseline-path."
        )
    return {row["question_id"]: row for row in read_jsonl(path) if "question_id" in row}


def resample_baseline(
    baseline_obj,
    client,
    condition,
    seed,
    tokenizer_model_id,
    baseline_path,
):
    q_id = baseline_obj["question_id"]
    reasoning_trace = extract_reasoning_trace(baseline_obj.get("response", ""))
    if not reasoning_trace:
        raise ValueError(f"Question {q_id} has no usable reasoning trace.")

    reasoning_token_ids = get_reasoning_token_ids(reasoning_trace, tokenizer_model_id)
    total_tokens = len(reasoning_token_ids)

    prompt = baseline_obj["prompt"]
    category = baseline_obj.get("category") or get_row_category(q_id)
    question = baseline_obj.get("question", ds[q_id]["question"])
    actual_answer_label = ds[q_id]["answer"]
    resample_results = []
    condition_token_ids, condition_metadata = build_resample_condition_full_token_ids(
        condition=condition,
        reasoning_token_ids=reasoning_token_ids,
        question_id=q_id,
        tokenizer_model_id=tokenizer_model_id,
        base_seed=seed,
    )

    for point in RESAMPLE_POINTS:
        prefix_token_count = decile_prefix_token_count(total_tokens, point)
        prefix_token_ids = condition_token_ids[:prefix_token_count]
        reasoning_prefix = decode_token_ids(prefix_token_ids, tokenizer_model_id)
        raw_probe_response, response = probe_answer(client, prompt, reasoning_prefix)
        resolution = resolve_answer_without_judge(response)
        answer = resolution["answer"]
        correct = answer == actual_answer_label
        resample_results.append(
            {
                "resample_point": point,
                "resample_tokens": prefix_token_count,
                "injected_prefix_text": reasoning_prefix,
                "response": response,
                "raw_probe_response": raw_probe_response,
                "judge_response": resolution["judge_response"],
                "direct_parsed_answer": resolution["direct_parsed_answer"],
                "judge_parsed_answer": resolution["judge_parsed_answer"],
                "used_llm_judge": resolution["used_llm_judge"],
                "answer_source": resolution["answer_source"],
                "parsed_answer": answer,
                "answer": answer,
                "correct": correct,
                "actual_answer": actual_answer_label,
                "actual_answer_label": actual_answer_label,
            }
        )

    return {
        "question_id": q_id,
        "category": category,
        "question": question,
        "prompt": prompt,
        "baseline_path": normalise_baseline_path(baseline_path),
        "condition": condition,
        "condition_metadata": condition_metadata,
        "resample_schema_version": OLLAMA_OPTIONS_RESAMPLE_SCHEMA_VERSION,
        "probe_method_version": OLLAMA_OPTIONS_PROBE_METHOD_VERSION,
        "model_id": client.model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "reasoning_token_count": total_tokens,
        "resample_results": resample_results,
    }


def iter_requested_qids(args):
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


def load_existing_result_rows(path):
    if not path.exists():
        return {}
    return {
        row["question_id"]: row
        for row in read_jsonl(path)
        if "question_id" in row
    }


def row_is_compatible(row, condition, tokenizer_model_id, model_id, baseline_path):
    points = row.get("resample_results", [])
    return (
        row.get("category") is not None
        and row.get("baseline_path") == normalise_baseline_path(baseline_path)
        and row.get("condition") == condition
        and row.get("resample_schema_version") == OLLAMA_OPTIONS_RESAMPLE_SCHEMA_VERSION
        and row.get("probe_method_version") == OLLAMA_OPTIONS_PROBE_METHOD_VERSION
        and row.get("model_id") == model_id
        and row.get("tokenizer_model_id") == tokenizer_model_id
        and len(points) == len(RESAMPLE_POINTS)
        and all(
            "resample_tokens" in point
            and "raw_probe_response" in point
            and "injected_prefix_text" in point
            and "judge_response" in point
            and "direct_parsed_answer" in point
            and "judge_parsed_answer" in point
            and "used_llm_judge" in point
            and "answer_source" in point
            and "response" in point
            and "parsed_answer" in point
            and "answer" in point
            for point in points
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample saved Ollama options CoTs at token-based deciles."
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
        default=len(ds),
        help="Exclusive end index within the selected category subset.",
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
        help="Recompute rows even if they already exist in the results file.",
    )
    parser.add_argument(
        "--condition",
        choices=SUPPORTED_RESAMPLE_CONDITIONS,
        default="original",
        help="Which intervention condition to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_CONTROL_DEFAULT_SEED,
        help="Base seed for stochastic control conditions.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=BASELINE_OPTIONS_PATH,
        help=f"Path to the Ollama baseline JSONL. Default: {BASELINE_OPTIONS_PATH}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional override for the output JSONL path.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=OLLAMA_MODEL_ID,
        help=f"Ollama model identifier used for probing. Default: {OLLAMA_MODEL_ID!r}.",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default=TOKENIZER_MODEL_ID,
        help=(
            "Hugging Face tokenizer identifier used for token-based deciles. "
            f"Default: {TOKENIZER_MODEL_ID!r}."
        ),
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
    if not get_category_qids(args.category):
        raise RuntimeError(
            f"No questions found for category={args.category!r}. "
            "Use 'all' to disable category filtering."
        )

    condition = validate_resample_condition(args.condition)
    output_path = args.output_path or get_options_results_path(condition)
    baselines = load_baselines(args.baseline_path)
    client = OllamaChatClient(
        model_id=args.model_id,
        base_url=args.ollama_base_url,
        timeout_seconds=args.ollama_timeout_seconds,
        keep_alive=args.ollama_keep_alive,
    )
    existing_rows = load_existing_result_rows(output_path)
    existing_ids = (
        set()
        if args.overwrite
        else {
            q_id
            for q_id, row in existing_rows.items()
            if row_is_compatible(
                row,
                condition,
                args.tokenizer_model_id,
                args.model_id,
                args.baseline_path,
            )
        }
    )

    for q_id in iter_requested_qids(args):
        if q_id < 0 or q_id >= len(ds):
            print(f"Skipping out-of-range question {q_id}.")
            continue
        if q_id in existing_ids:
            print(f"Skipping question {q_id}; resample already exists.")
            continue

        baseline_obj = baselines.get(q_id)
        if baseline_obj is None:
            print(f"Skipping question {q_id}; no baseline row found.")
            continue
        if not category_matches(q_id, args.category):
            print(
                f"Skipping question {q_id}; category={get_row_category(q_id)!r} "
                f"does not match requested category={args.category!r}."
            )
            continue

        print(f"Resampling question {q_id} for condition={condition}...")
        result_obj = resample_baseline(
            baseline_obj,
            client=client,
            condition=condition,
            seed=args.seed,
            tokenizer_model_id=args.tokenizer_model_id,
            baseline_path=args.baseline_path,
        )
        existing_rows[q_id] = result_obj
        write_jsonl(
            output_path,
            [existing_rows[key] for key in sorted(existing_rows)],
        )
        existing_ids.add(q_id)
        print(f"Saved resample for question {q_id} to {output_path}.")


if __name__ == "__main__":
    main()
