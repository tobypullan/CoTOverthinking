import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from groq import Groq


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import (
    BASELINE_OPTIONS_PATH,
    MODEL_ID,
    OPTIONS_RESULTS_PATH,
    PROBE_MAX_COMPLETION_TOKENS,
    PROBE_METHOD_VERSION,
    PROBE_TEMPERATURE,
    PROBE_TOP_P,
    build_probe_messages,
    decile_prefix_token_count,
    extract_final_answer_text,
    extract_reasoning_trace,
    get_reasoning_token_count,
    judge_answer_label,
    probe_response_needs_retry,
    read_jsonl,
    truncate_reasoning_trace,
    write_jsonl,
)


ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

RESAMPLE_POINTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def load_baselines():
    return {row["question_id"]: row for row in read_jsonl(BASELINE_OPTIONS_PATH)}


def probe_answer(prompt, reasoning_prefix):
    attempts = (False, True)
    last_raw_probe_response = ""
    last_response = ""

    for retry in attempts:
        completion = client.chat.completions.create(
            messages=build_probe_messages(prompt, reasoning_prefix, retry=retry),
            model=MODEL_ID,
            temperature=PROBE_TEMPERATURE,
            top_p=PROBE_TOP_P,
            max_completion_tokens=PROBE_MAX_COMPLETION_TOKENS,
        )
        raw_probe_response = completion.choices[0].message.content or ""
        response = extract_final_answer_text(raw_probe_response)

        last_raw_probe_response = raw_probe_response
        last_response = response
        if not probe_response_needs_retry(raw_probe_response, response):
            break

    return last_raw_probe_response, last_response


def resample_baseline(baseline_obj):
    q_id = baseline_obj["question_id"]
    reasoning_trace = baseline_obj.get("reasoning_trace")
    if not reasoning_trace:
        reasoning_trace = extract_reasoning_trace(baseline_obj.get("response", ""))
    if not reasoning_trace:
        raise ValueError(f"Question {q_id} has no usable reasoning trace.")

    total_tokens = baseline_obj.get("reasoning_token_count")
    if total_tokens is None:
        total_tokens = get_reasoning_token_count(reasoning_trace)

    prompt = baseline_obj["prompt"]
    options = ds[q_id]["options"]
    actual_answer = ds[q_id]["answer"]
    resample_results = []

    for point in RESAMPLE_POINTS:
        prefix_token_count = decile_prefix_token_count(total_tokens, point)
        reasoning_prefix = truncate_reasoning_trace(reasoning_trace, prefix_token_count)
        raw_probe_response, response = probe_answer(prompt, reasoning_prefix)
        judge_response, answer = judge_answer_label(client, response, options)
        correct = answer == actual_answer
        resample_results.append(
            {
                "resample_point": point,
                "resample_tokens": prefix_token_count,
                "response": response,
                "raw_probe_response": raw_probe_response,
                "judge_response": judge_response,
                "parsed_answer": f"ANSWER: {answer}" if answer != "UNKNOWN" else "UNKNOWN",
                "answer": answer,
                "correct": correct,
                "actual_answer": actual_answer,
            }
        )

    return {
        "question_id": q_id,
        "question": baseline_obj.get("question", ds[q_id]["question"]),
        "probe_method_version": PROBE_METHOD_VERSION,
        "reasoning_token_count": total_tokens,
        "resample_results": resample_results,
    }


def iter_requested_qids(args):
    if args.question_ids:
        return args.question_ids
    return range(args.start, args.end)


def load_existing_result_rows():
    if not OPTIONS_RESULTS_PATH.exists():
        return {}
    return {
        row["question_id"]: row
        for row in read_jsonl(OPTIONS_RESULTS_PATH)
        if "question_id" in row
    }


def row_is_compatible(row):
    points = row.get("resample_results", [])
    return (
        row.get("probe_method_version") == PROBE_METHOD_VERSION
        and len(points) == len(RESAMPLE_POINTS)
        and all(
        "resample_tokens" in point and "raw_probe_response" in point
        for point in points
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample saved Qwen3-32B options CoTs at token-based deciles."
    )
    parser.add_argument("--start", type=int, default=0, help="Inclusive start qid.")
    parser.add_argument(
        "--end",
        type=int,
        default=len(ds),
        help="Exclusive end qid.",
    )
    parser.add_argument(
        "--question-ids",
        type=int,
        nargs="+",
        help="Specific question ids to run instead of a contiguous range.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows even if they already exist in the results file.",
    )
    return parser.parse_args()


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    args = parse_args()
    baselines = load_baselines()
    existing_rows = load_existing_result_rows()
    existing_ids = (
        set()
        if args.overwrite
        else {q_id for q_id, row in existing_rows.items() if row_is_compatible(row)}
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
        if baseline_obj.get("complete_reason") != "stop":
            print(
                f"Skipping question {q_id}; baseline finish_reason="
                f"{baseline_obj.get('complete_reason')}"
            )
            continue

        print(f"Resampling question {q_id}...")
        result_obj = resample_baseline(baseline_obj)
        existing_rows[q_id] = result_obj
        write_jsonl(
            OPTIONS_RESULTS_PATH,
            [existing_rows[key] for key in sorted(existing_rows)],
        )
        existing_ids.add(q_id)
        print(f"Saved resample for question {q_id}.")


if __name__ == "__main__":
    main()
