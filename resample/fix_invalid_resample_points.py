import argparse
import os
import sys
from pathlib import Path

from groq import Groq


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import (
    BASELINE_OPTIONS_PATH,
    RANDOM_CONTROL_DEFAULT_SEED,
    SUPPORTED_RESAMPLE_CONDITIONS,
    build_resample_condition_full_token_ids,
    decode_token_ids,
    decile_prefix_token_count,
    extract_options_from_prompt,
    extract_reasoning_trace,
    get_options_results_path,
    get_reasoning_token_ids,
    probe_answer,
    read_jsonl,
    resolve_answer_label,
    validate_resample_condition,
    write_jsonl,
)


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def load_baselines():
    return {row["question_id"]: row for row in read_jsonl(BASELINE_OPTIONS_PATH)}


def point_needs_fix(point):
    answer = str(point.get("answer", "")).strip().upper()
    return answer in {"", "UNKNOWN", "INVALID"}


def prompt_for_acceptance(candidate):
    print("\nCandidate response:")
    print(candidate)

    while True:
        decision = input("Accept this response? [y/n]: ").strip().lower()
        if decision in {"y", "n"}:
            return decision == "y"
        print("Please enter 'y' or 'n'.")


def repair_point(question_id, point, baseline_obj, options, condition, seed):
    reasoning_trace = baseline_obj.get("reasoning_trace")
    if not reasoning_trace:
        reasoning_trace = extract_reasoning_trace(baseline_obj.get("response", ""))
    if not reasoning_trace:
        raise ValueError(f"Question {question_id} has no usable reasoning trace.")

    reasoning_token_ids = get_reasoning_token_ids(reasoning_trace)
    total_tokens = baseline_obj.get("reasoning_token_count")
    if total_tokens is None:
        total_tokens = len(reasoning_token_ids)
    total_tokens = min(int(total_tokens), len(reasoning_token_ids))
    reasoning_token_ids = reasoning_token_ids[:total_tokens]

    resample_point = point["resample_point"]
    prefix_token_count = decile_prefix_token_count(total_tokens, resample_point)
    condition_token_ids, _ = build_resample_condition_full_token_ids(
        condition=condition,
        reasoning_token_ids=reasoning_token_ids,
        question_id=question_id,
        base_seed=seed,
    )
    reasoning_prefix = decode_token_ids(condition_token_ids[:prefix_token_count])

    print(f"\nQuestion {question_id} | resample point {resample_point}")
    print("\nCurrent saved response:")
    print(point.get("response", ""))

    while True:
        raw_probe_response, response = probe_answer(
            client,
            baseline_obj["prompt"],
            reasoning_prefix,
        )
        if not prompt_for_acceptance(response):
            continue

        resolution = resolve_answer_label(client, response, options)
        judge_response = resolution["judge_response"]
        answer = resolution["answer"]
        updated_point = dict(point)
        updated_point["resample_tokens"] = prefix_token_count
        updated_point["injected_prefix_text"] = reasoning_prefix
        updated_point["response"] = response
        updated_point["raw_probe_response"] = raw_probe_response
        updated_point["judge_response"] = judge_response
        updated_point["direct_parsed_answer"] = resolution["direct_parsed_answer"]
        updated_point["judge_parsed_answer"] = resolution["judge_parsed_answer"]
        updated_point["used_llm_judge"] = resolution["used_llm_judge"]
        updated_point["answer_source"] = resolution["answer_source"]
        updated_point["parsed_answer"] = (
            f"ANSWER: {answer}" if answer != "UNKNOWN" else "UNKNOWN"
        )
        updated_point["answer"] = answer
        updated_point["correct"] = answer == str(updated_point["actual_answer"]).strip().upper()
        return updated_point


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair invalid options resample points for a given condition."
    )
    parser.add_argument(
        "--condition",
        choices=SUPPORTED_RESAMPLE_CONDITIONS,
        default="original",
        help="Which intervention condition to repair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_CONTROL_DEFAULT_SEED,
        help="Base seed for stochastic control conditions.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional override for the results JSONL path.",
    )
    return parser.parse_args()


def fix_invalid_points():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    args = parse_args()
    condition = validate_resample_condition(args.condition)
    results_path = args.output_path or get_options_results_path(condition)
    results = list(read_jsonl(results_path))
    baselines = load_baselines()

    fixed_count = 0
    total_invalid = 0

    for segment in results:
        question_id = segment["question_id"]
        baseline_obj = baselines.get(question_id)
        if baseline_obj is None:
            print(f"Skipping question {question_id}: no baseline found.")
            continue

        try:
            option_texts = extract_options_from_prompt(baseline_obj["prompt"])
        except ValueError:
            print(f"Skipping question {question_id}: prompt is malformed.")
            continue

        for idx, point in enumerate(segment.get("resample_results", [])):
            if not point_needs_fix(point):
                continue

            total_invalid += 1
            updated_point = repair_point(
                question_id,
                point,
                baseline_obj,
                option_texts,
                condition=condition,
                seed=args.seed,
            )
            segment["resample_results"][idx] = updated_point
            write_jsonl(results_path, results)
            print(
                f"Saved updated result for question {question_id} at point "
                f"{point['resample_point']}."
            )
            if not point_needs_fix(updated_point):
                fixed_count += 1

    print(f"\nFinished. Updated {fixed_count} of {total_invalid} invalid resample points.")


if __name__ == "__main__":
    fix_invalid_points()
