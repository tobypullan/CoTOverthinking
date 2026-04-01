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
    FULL_TRACE_MAX_COMPLETION_TOKENS,
    FULL_TRACE_TEMPERATURE,
    FULL_TRACE_TOP_P,
    MODEL_ID,
    OPEN_BASELINE_SCHEMA_VERSION,
    OPEN_DEFAULT_DATASET_CATEGORY,
    OPEN_SCORING_MODE,
    OPEN_SYSTEM_PROMPT,
    build_open_prompt,
    extract_final_answer_text,
    extract_reasoning_trace,
    get_reasoning_token_count,
    read_jsonl,
    resolve_open_answer_correctness,
    write_jsonl,
)


ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


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


def baseline_cot(q_id):
    question = ds[q_id]["question"]
    options = ds[q_id]["options"]
    category = get_row_category(q_id)
    actual_answer_label = ds[q_id]["answer"]
    actual_answer = options[ord(actual_answer_label) - 65]
    prompt = build_open_prompt(question)
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": OPEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=MODEL_ID,
        temperature=FULL_TRACE_TEMPERATURE,
        top_p=FULL_TRACE_TOP_P,
        max_completion_tokens=FULL_TRACE_MAX_COMPLETION_TOKENS,
    )

    raw_response = completion.choices[0].message.content or ""
    finish_reason = completion.choices[0].finish_reason
    reasoning_trace = extract_reasoning_trace(raw_response)
    reasoning_token_count = (
        get_reasoning_token_count(reasoning_trace)
        if reasoning_trace is not None
        else None
    )

    final_answer_text = ""
    answer_candidate = ""
    judge_response = None
    judge_finish_reason = None
    judge_attempt_count = 0
    direct_verdict = None
    judge_verdict = None
    used_llm_judge = False
    verdict_source = "unknown"
    verdict = "UNKNOWN"
    correct = False
    if finish_reason == "stop":
        final_answer_text = extract_final_answer_text(raw_response)
        resolution = resolve_open_answer_correctness(
            client,
            question,
            final_answer_text,
            actual_answer,
            raw_response=raw_response,
        )
        judge_response = resolution["judge_response"]
        judge_finish_reason = resolution["judge_finish_reason"]
        judge_attempt_count = resolution["judge_attempt_count"]
        answer_candidate = resolution["answer_candidate"]
        direct_verdict = resolution["direct_verdict"]
        judge_verdict = resolution["judge_verdict"]
        used_llm_judge = resolution["used_llm_judge"]
        verdict_source = resolution["verdict_source"]
        verdict = resolution["verdict"]
        correct = resolution["correct"]
    return {
        "question_id": q_id,
        "category": category,
        "question": question,
        "response": raw_response,
        "reasoning_trace": reasoning_trace,
        "reasoning_token_count": reasoning_token_count,
        "final_answer_text": final_answer_text,
        "answer_candidate": answer_candidate,
        "judge_response": judge_response,
        "judge_finish_reason": judge_finish_reason,
        "judge_attempt_count": judge_attempt_count,
        "direct_verdict": direct_verdict,
        "judge_verdict": judge_verdict,
        "used_llm_judge": used_llm_judge,
        "verdict_source": verdict_source,
        "verdict": verdict,
        "correct": correct,
        "actual_answer": actual_answer,
        "actual_answer_label": actual_answer_label,
        "baseline_schema_version": OPEN_BASELINE_SCHEMA_VERSION,
        "scoring_mode": OPEN_SCORING_MODE,
        "complete_reason": finish_reason,
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


def load_existing_baseline_rows():
    if not BASELINE_OPEN_PATH.exists():
        return {}
    return {
        row["question_id"]: row
        for row in read_jsonl(BASELINE_OPEN_PATH)
        if "question_id" in row
    }


def row_is_compatible(row):
    return (
        row.get("baseline_schema_version") == OPEN_BASELINE_SCHEMA_VERSION
        and row.get("scoring_mode") == OPEN_SCORING_MODE
        and "category" in row
        and "question" in row
        and "reasoning_trace" in row
        and "reasoning_token_count" in row
        and "final_answer_text" in row
        and "answer_candidate" in row
        and "judge_finish_reason" in row
        and "judge_attempt_count" in row
        and "direct_verdict" in row
        and "judge_verdict" in row
        and "verdict" in row
        and "verdict_source" in row
        and "actual_answer_label" in row
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline Qwen3-32B CoTs for MMLU-Pro open-answer prompts."
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
        help="Regenerate rows even if they already exist in the baseline file.",
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
    existing_rows = load_existing_baseline_rows()
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
            print(f"Skipping question {q_id}; baseline already exists.")
            continue

        print(f"Processing question {q_id}...")
        baseline_row = baseline_cot(q_id)
        existing_rows[q_id] = baseline_row
        write_jsonl(
            BASELINE_OPEN_PATH,
            [existing_rows[key] for key in sorted(existing_rows)],
        )
        existing_ids.add(q_id)
        print(
            f"Saved question {q_id}. finish_reason={baseline_row['complete_reason']} "
            f"verdict={baseline_row['verdict']} correct={baseline_row['correct']}"
        )


if __name__ == "__main__":
    main()
