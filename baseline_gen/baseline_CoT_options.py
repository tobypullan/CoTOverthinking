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
    FULL_TRACE_MAX_COMPLETION_TOKENS,
    FULL_TRACE_TEMPERATURE,
    FULL_TRACE_TOP_P,
    MODEL_ID,
    OPTIONS_BASELINE_SCHEMA_VERSION,
    OPTIONS_DEFAULT_DATASET_CATEGORY,
    SYSTEM_PROMPT,
    build_options_prompt,
    category_value_matches,
    extract_final_answer_text,
    extract_reasoning_trace,
    get_reasoning_token_count,
    read_jsonl,
    resolve_answer_label,
    write_jsonl,
)


ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def get_row_category(q_id):
    return ds[q_id].get("category")


def category_matches(q_id, category):
    return category_value_matches(get_row_category(q_id), category)


def get_category_qids(category):
    return [q_id for q_id in range(len(ds)) if category_matches(q_id, category)]


def baseline_cot(q_id):
    question = ds[q_id]["question"]
    options = ds[q_id]["options"]
    category = get_row_category(q_id)
    prompt = build_options_prompt(question, options)
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    judge_response = None
    direct_parsed_answer = "UNKNOWN"
    judge_parsed_answer = None
    used_llm_judge = False
    answer_source = "unknown"
    answer = "UNKNOWN"
    if finish_reason == "stop":
        final_answer_text = extract_final_answer_text(raw_response)
        resolution = resolve_answer_label(
            client,
            final_answer_text,
            options,
        )
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


def load_existing_baseline_rows():
    if not BASELINE_OPTIONS_PATH.exists():
        return {}
    return {
        row["question_id"]: row
        for row in read_jsonl(BASELINE_OPTIONS_PATH)
        if "question_id" in row
    }


def row_is_compatible(row):
    return (
        row.get("baseline_schema_version") == OPTIONS_BASELINE_SCHEMA_VERSION
        and row.get("category") is not None
        and "reasoning_token_count" in row
        and "final_answer_text" in row
        and "answer" in row
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline Qwen3-32B CoTs for MMLU-Pro options prompts."
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
            BASELINE_OPTIONS_PATH,
            [existing_rows[key] for key in sorted(existing_rows)],
        )
        existing_ids.add(q_id)
        print(
            f"Saved question {q_id}. finish_reason={baseline_row['complete_reason']} "
            f"answer={baseline_row['answer']} correct={baseline_row['correct']}"
        )


if __name__ == "__main__":
    main()
