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
    SYSTEM_PROMPT,
    build_options_prompt,
    extract_final_answer_text,
    extract_reasoning_trace,
    get_reasoning_token_count,
    judge_answer_label,
    read_jsonl,
    write_jsonl,
)


ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def baseline_cot(q_id):
    question = ds[q_id]["question"]
    options = ds[q_id]["options"]
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
    answer = "UNKNOWN"
    if finish_reason == "stop":
        final_answer_text = extract_final_answer_text(raw_response)
        judge_response, answer = judge_answer_label(
            client,
            final_answer_text,
            options,
        )

    correct = answer == ds[q_id]["answer"]
    return {
        "question_id": q_id,
        "question": question,
        "prompt": prompt,
        "response": raw_response,
        "reasoning_trace": reasoning_trace,
        "reasoning_token_count": reasoning_token_count,
        "final_answer_text": final_answer_text,
        "judge_response": judge_response,
        "answer": answer,
        "correct": correct,
        "actual_answer": ds[q_id]["answer"],
        "complete_reason": finish_reason,
    }


def iter_requested_qids(args):
    if args.question_ids:
        return args.question_ids
    return range(args.start, args.end)


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
        "reasoning_token_count" in row
        and "final_answer_text" in row
        and "answer" in row
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline Qwen3-32B CoTs for MMLU-Pro options prompts."
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
        help="Regenerate rows even if they already exist in the baseline file.",
    )
    return parser.parse_args()


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    args = parse_args()
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
