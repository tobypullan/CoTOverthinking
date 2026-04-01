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

from open_experiment_utils import (
    BASELINE_OPEN_PATH,
    OPEN_DEFAULT_DATASET_CATEGORY,
    OPEN_PROBE_METHOD_VERSION,
    OPEN_RESAMPLE_SCHEMA_VERSION,
    OPEN_SCORING_MODE,
    RANDOM_CONTROL_DEFAULT_SEED,
    SUPPORTED_RESAMPLE_CONDITIONS,
    build_resample_condition_full_token_ids,
    decode_token_ids,
    decile_prefix_token_count,
    extract_reasoning_trace,
    get_open_results_path,
    get_reasoning_token_ids,
    probe_open_answer,
    read_jsonl,
    resolve_open_answer_correctness,
    validate_resample_condition,
    write_jsonl,
)


ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

RESAMPLE_POINTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def normalise_category(category):
    return (category or "").strip().casefold()


def get_row_category(q_id):
    return ds[q_id].get("category")


def category_matches(q_id, category):
    if normalise_category(category) == "all":
        return True
    return normalise_category(get_row_category(q_id)) == normalise_category(category)


def get_category_qids(category):
    return [q_id for q_id in range(len(ds)) if category_matches(q_id, category)]


def load_baselines():
    return {row["question_id"]: row for row in read_jsonl(BASELINE_OPEN_PATH)}


def resample_baseline(baseline_obj, condition, seed):
    q_id = baseline_obj["question_id"]
    reasoning_trace = baseline_obj.get("reasoning_trace")
    if not reasoning_trace:
        reasoning_trace = extract_reasoning_trace(baseline_obj.get("response", ""))
    if not reasoning_trace:
        raise ValueError(f"Question {q_id} has no usable reasoning trace.")

    reasoning_token_ids = get_reasoning_token_ids(reasoning_trace)
    total_tokens = baseline_obj.get("reasoning_token_count")
    if total_tokens is None:
        total_tokens = len(reasoning_token_ids)
    total_tokens = min(int(total_tokens), len(reasoning_token_ids))
    reasoning_token_ids = reasoning_token_ids[:total_tokens]

    question = baseline_obj.get("question", ds[q_id]["question"])
    category = baseline_obj.get("category") or get_row_category(q_id)
    options = ds[q_id]["options"]
    actual_answer_label = ds[q_id]["answer"]
    actual_answer = options[ord(actual_answer_label) - 65]
    resample_results = []
    condition_token_ids, condition_metadata = build_resample_condition_full_token_ids(
        condition=condition,
        reasoning_token_ids=reasoning_token_ids,
        question_id=q_id,
        base_seed=seed,
    )

    for point in RESAMPLE_POINTS:
        prefix_token_count = decile_prefix_token_count(total_tokens, point)
        prefix_token_ids = condition_token_ids[:prefix_token_count]
        reasoning_prefix = decode_token_ids(prefix_token_ids)
        raw_probe_response, response = probe_open_answer(client, question, reasoning_prefix)
        resolution = resolve_open_answer_correctness(
            client,
            question,
            response,
            actual_answer,
            raw_response=raw_probe_response,
        )
        resample_results.append(
            {
                "resample_point": point,
                "resample_tokens": prefix_token_count,
                "injected_prefix_text": reasoning_prefix,
                "response": response,
                "raw_probe_response": raw_probe_response,
                "judge_response": resolution["judge_response"],
                "judge_finish_reason": resolution["judge_finish_reason"],
                "judge_attempt_count": resolution["judge_attempt_count"],
                "answer_candidate": resolution["answer_candidate"],
                "direct_verdict": resolution["direct_verdict"],
                "judge_verdict": resolution["judge_verdict"],
                "used_llm_judge": resolution["used_llm_judge"],
                "verdict_source": resolution["verdict_source"],
                "verdict": resolution["verdict"],
                "correct": resolution["correct"],
                "actual_answer": actual_answer,
                "actual_answer_label": actual_answer_label,
            }
        )

    return {
        "question_id": q_id,
        "category": category,
        "question": question,
        "condition": condition,
        "condition_metadata": condition_metadata,
        "resample_schema_version": OPEN_RESAMPLE_SCHEMA_VERSION,
        "probe_method_version": OPEN_PROBE_METHOD_VERSION,
        "reasoning_token_count": total_tokens,
        "scoring_mode": OPEN_SCORING_MODE,
        "resample_results": resample_results,
    }


def iter_requested_qids(args):
    if args.question_ids:
        requested = []
        for q_id in args.question_ids:
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


def row_is_compatible(row, condition):
    points = row.get("resample_results", [])
    return (
        row.get("category") is not None
        and
        row.get("condition") == condition
        and row.get("resample_schema_version") == OPEN_RESAMPLE_SCHEMA_VERSION
        and row.get("probe_method_version") == OPEN_PROBE_METHOD_VERSION
        and row.get("scoring_mode") == OPEN_SCORING_MODE
        and len(points) == len(RESAMPLE_POINTS)
        and all(
            "resample_tokens" in point
            and "raw_probe_response" in point
            and "injected_prefix_text" in point
            and "judge_finish_reason" in point
            and "judge_attempt_count" in point
            and "answer_candidate" in point
            and "direct_verdict" in point
            and "judge_verdict" in point
            and "used_llm_judge" in point
            and "verdict_source" in point
            and "verdict" in point
            and "response" in point
            for point in points
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample saved Qwen3-32B open-answer CoTs at token-based deciles."
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
        default=OPEN_DEFAULT_DATASET_CATEGORY,
        help=(
            "Dataset category to run. Use 'all' to disable category filtering. "
            f"Default: {OPEN_DEFAULT_DATASET_CATEGORY}."
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
        "--output-path",
        type=Path,
        help="Optional override for the output JSONL path.",
    )
    return parser.parse_args()


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    args = parse_args()
    if not get_category_qids(args.category):
        raise RuntimeError(
            f"No questions found for category={args.category!r}. "
            "Use 'all' to disable category filtering."
        )
    condition = validate_resample_condition(args.condition)
    output_path = args.output_path or get_open_results_path(condition)
    baselines = load_baselines()
    existing_rows = load_existing_result_rows(output_path)
    existing_ids = (
        set()
        if args.overwrite
        else {
            q_id
            for q_id, row in existing_rows.items()
            if row_is_compatible(row, condition)
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
        if baseline_obj.get("complete_reason") != "stop":
            print(
                f"Skipping question {q_id}; baseline finish_reason="
                f"{baseline_obj.get('complete_reason')}"
            )
            continue

        print(f"Resampling question {q_id} for condition={condition}...")
        result_obj = resample_baseline(
            baseline_obj,
            condition=condition,
            seed=args.seed,
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
