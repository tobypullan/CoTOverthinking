import json
import os
import re
from pathlib import Path

from groq import Groq


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR.parent / "resample_results" / "options_results.jsonl"
BASELINE_PATH = SCRIPT_DIR.parent / "baseline" / "baseline_CoTs_options.jsonl"

LABELS = set("ABCDEFGHIJ")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


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
    if stripped in LABELS:
        return stripped

    return None


def point_needs_fix(point):
    return str(point.get("answer", "")).strip().lower() == "invalid"


def truncate_cot(response, resample_length):
    truncated = response[:resample_length]
    last_period = truncated.rfind(".")
    if last_period != -1:
        return truncated[: last_period + 1]
    return truncated


def resample_response(response, resample_length, prompt):
    resampled_reasoning = truncate_cot(response, resample_length)
    last_content = ""

    for _ in range(5):
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You must answer the question based on the incomplete reasoning "
                        "with a single token. Based on this reasoning alone, you must "
                        "choose an answer label option. You may not produce any tokens "
                        "other than the option label token. Answer with a single label "
                        "token: the label of the option you choose. For example, if you "
                        "choose option A, answer with 'A' and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"/no_think question: {prompt}, reasoning: "
                        f"{resampled_reasoning}\n\nAnswer Label: "
                    ),
                },
            ],
            model="qwen/qwen3-32b",
        )
        last_content = completion.choices[0].message.content.strip()
        if len(last_content) < 200:
            break

    return last_content


def load_baselines():
    baselines = {}
    for row in read_jsonl(BASELINE_PATH):
        baselines[row["question_id"]] = row
    return baselines


def prompt_for_acceptance(candidate):
    print("\nCandidate response:")
    print(candidate)

    while True:
        decision = input("Accept this response? [y/n]: ").strip().lower()
        if decision in {"y", "n"}:
            return decision == "y"
        print("Please enter 'y' or 'n'.")


def repair_point(question_id, question_text, point, baseline_obj):
    baseline_response = baseline_obj["response"]
    baseline_length = len(baseline_response)
    resample_point = point["resample_point"]
    resample_length = int(baseline_length * resample_point)

    print(f"\nQuestion {question_id} | resample point {resample_point}")
    if question_text:
        print(question_text)
    print("\nCurrent saved response:")
    print(point.get("response", ""))

    while True:
        candidate = resample_response(
            baseline_response,
            resample_length,
            baseline_obj["prompt"],
        )

        if not prompt_for_acceptance(candidate):
            continue

        updated_point = dict(point)
        updated_point["response"] = candidate

        label = extract_answer_label(candidate)
        if label:
            updated_point["parsed_answer"] = f"ANSWER: {label}"
            updated_point["answer"] = label
            actual_answer = str(updated_point.get("actual_answer", "")).strip().upper()
            if actual_answer in LABELS:
                updated_point["correct"] = label == actual_answer
        else:
            updated_point["parsed_answer"] = "invalid"
            updated_point["answer"] = "invalid"
            print("Accepted response did not contain a parseable label; kept it marked invalid.")

        return updated_point


def fix_invalid_points():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    results = list(read_jsonl(RESULTS_PATH))
    baselines = load_baselines()

    fixed_count = 0
    total_invalid = 0

    for segment in results:
        question_id = segment["question_id"]
        baseline_obj = baselines.get(question_id)

        if baseline_obj is None:
            print(f"Skipping question {question_id}: no baseline found.")
            continue

        for idx, point in enumerate(segment.get("resample_results", [])):
            if not point_needs_fix(point):
                continue

            total_invalid += 1
            updated_point = repair_point(
                question_id,
                segment.get("question", ""),
                point,
                baseline_obj,
            )
            segment["resample_results"][idx] = updated_point
            write_jsonl(RESULTS_PATH, results)
            print(f"Saved updated result for question {question_id} at point {point['resample_point']}.")
            if not point_needs_fix(updated_point):
                fixed_count += 1

    print(f"\nFinished. Updated {fixed_count} of {total_invalid} invalid resample points.")


if __name__ == "__main__":
    fix_invalid_points()
