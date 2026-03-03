import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from groq import Groq

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_PATH = SCRIPT_DIR.parent / "baseline" / "baseline_CoTs_open.jsonl"
RESULTS_PATH = SCRIPT_DIR.parent / "resample_results" / "open_results.jsonl"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def extract_answer_label(text):
    if not text:
        return None

    text = text.strip().rstrip("*_`")
    patterns = [
        r"CHOSEN OPTION:\s*\[?([A-J])\]?\b",
        r"ANSWER:\s*\[?([A-J])\]?\b",
        r"\banswer\s*(?:is|:)\s*\[?([A-J])\]?\b",
        r"\boption\s+([A-J])\b",
        r"\b([A-J])\b[\s\]\).,:;!?*_`-]*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def truncate_CoT(response, resample_length):
    """Truncate the CoT to a sentence boundary close to the requested length."""
    truncated = response[:resample_length]
    last_sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_sentence_end != -1:
        return truncated[: last_sentence_end + 1]
    return truncated


def resample_response(response, resample_length, question):
    """Generate an open-form answer from truncated reasoning."""
    resampled_reasoning = truncate_CoT(response, resample_length)
    resampled_answer = ""
    tries = 0

    while tries < 5:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You must answer the question using only the incomplete reasoning provided. Do not continue the reasoning. Return a concise final answer only.",
                },
                {
                    "role": "user",
                    "content": f"/no_think question: {question}\n\nIncomplete reasoning: {resampled_reasoning}\n\nFinal answer:",
                },
            ],
            model="qwen/qwen3-32b",
        )
        resampled_answer = completion.choices[0].message.content.strip()
        tries += 1
        if len(resampled_answer) < 1000:
            break

    return resampled_answer


def model_judge(response, options):
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are given a model's response to a question. You must decide which labelled option the response best matches. Do not choose the answer you think is correct independently of the response. Explain briefly, then end with the single label you chose.",
            },
            {
                "role": "user",
                "content": f"Response: {response}\n\nOptions:\n{options}",
            },
        ],
        model="openai/gpt-oss-20b",
    )
    return completion.choices[0].message.content.strip()


def judge_and_parse(response, options):
    judge_response = None

    for _ in range(5):
        judge_response = model_judge(response, options)
        parsed_answer = extract_answer_label(judge_response)
        if parsed_answer:
            return judge_response, parsed_answer

    return judge_response, "UNKNOWN"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e


def resample(baseline_obj):
    """Resample one open-question CoT across 0%-100% truncation points."""
    q_id = baseline_obj["question_id"]
    question = baseline_obj["question"]
    labelled_options = "\n".join(
        f"{chr(65 + i)}: {option}" for i, option in enumerate(ds[q_id]["options"])
    )
    actual_answer_label = ds[q_id]["answer"]
    actual_answer = ds[q_id]["options"][ord(actual_answer_label) - 65]
    baseline_length = len(baseline_obj["response"])
    resample_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    resample_results = []

    for point in resample_points:
        resample_length = int(baseline_length * point)
        resampled = resample_response(
            baseline_obj["response"],
            resample_length,
            question,
        )
        judge_response, parsed_answer = judge_and_parse(resampled, labelled_options)
        correct = parsed_answer == actual_answer_label
        resample_results.append(
            {
                "resample_point": point,
                "response": resampled,
                "judge_response": judge_response,
                "parsed_answer": parsed_answer,
                "correct": correct,
                "actual_answer": actual_answer,
                "actual_answer_label": actual_answer_label,
            }
        )

    result_obj = {
        "question_id": q_id,
        "question": question,
        "resample_results": resample_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_obj) + "\n")

    print(f"Resampled question {q_id}")

for baseline_obj in read_jsonl(BASELINE_PATH):
    q_id = baseline_obj["question_id"]
    if baseline_obj["complete_reason"] == "stop":
        print(f"Resampling question {q_id}...")
        resample(baseline_obj)
    else:
        print(f"Skipping question {q_id} because baseline CoT was not complete.")

