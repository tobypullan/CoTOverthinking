import os
from groq import Groq
from datasets import load_dataset
import json
import re

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def extract_answer_label(text):
    if not text:
        return None
    text = text.strip()
    patterns = [
        r"ANSWER:\s*\[?([A-J])\]?\b",
        r"\banswer\s*(?:is|:)\s*\[?([A-J])\]?\b",
        r"\boption\s+([A-J])\b",
        r"\b([A-J])\b[\s\]\).,:;!?]*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None

def truncate_CoT(response, resample_length):
    """Truncates the CoT response to the given length, ensuring that the truncated response ends at a complete sentence."""
    truncated = response[:resample_length]
    last_period = truncated.rfind(".")
    if last_period != -1:
        return truncated[:last_period + 1]
    else:
        return truncated

def resample_response(response, resample_length, prompt):
    """Resamples the response to the given length, ensuring that the resampled response ends at a complete sentence."""
    resampled = truncate_CoT(response, resample_length)
    response_length = 10000
    tries = 0
    while tries < 5:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You must answer the question based on the incomplete reasoning with a **single** token. Based on this reasoning alone, you must choose an answer label option. You may not produce any tokens other than the option label token. Answer with a **single** label token - the label of the option you choose. For example, if you choose option A, answer with 'A' and nothing else. In no scenario should you produce any token other than an answer label."
                },
                {
                    "role": "user",
                    "content": f"/no_think question: {prompt}, reasoning: {resampled} \n\nAnswer Label: "
                }
            ],
            model="qwen/qwen3-32b",
        )
        response_length = len(response.choices[0].message.content.strip())
        tries += 1
        if response_length < 200:
            break
    return response.choices[0].message.content.strip()

def model_answer_parser(response, options):
    direct_label = extract_answer_label(response)
    if direct_label:
        return f"ANSWER: {direct_label}"

    labelled_options = [f"{chr(65+i)}: {options[i]}" for i in range(len(options))]
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You extract the answer label already chosen in another model's response. Do not solve the question yourself. You may use the provided options only to map quoted or paraphrased option text back to a label. If the response does not clearly choose exactly one option, return only UNKNOWN."
            },
            {
                "role": "user",
                "content": f"Options: {labelled_options}\n\nModel response: {response}\n\nReturn only one token: A, B, C, D, E, F, G, H, I, J, or UNKNOWN.",
            }
        ],
        model="llama-3.1-8b-instant",
    )
    parsed_response = completion.choices[0].message.content.strip()
    parsed_label = extract_answer_label(parsed_response)
    return f"ANSWER: {parsed_label}" if parsed_label else "UNKNOWN"

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

def resample(q_id):
    """Resamples the baseline options CoT of q_id
        - writes result object to resample_results/options_results.jsonl
        - result object contains 0%, 10%, ..., 100% results
            - the output at given point
            - whether answer is correct based on llm parser
            - the correct answer
    """
    for line in read_jsonl("../baseline/baseline_CoTs_options.jsonl"):
        if line["question_id"] == q_id:
            baseline_obj = line
            print(line)
            break
    question = ds[q_id]["question"]
    baseline_length = len(baseline_obj["response"])
    resample_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    resample_results = []
    for point in resample_points:
        resample_length = int(baseline_length * point)
        resampled = resample_response(baseline_obj["response"], resample_length, baseline_obj["prompt"])
        pre_answer = model_answer_parser(resampled, ds[q_id]["options"])
        match = re.search(r"ANSWER:\s*([A-J])", pre_answer)
        parsed_answer = match.group(1) if match else "UNKNOWN"
        correct = parsed_answer == ds[q_id]["answer"]
        resample_results.append({
            "resample_point": point,
            "response": resampled,
            "parsed_answer": pre_answer,
            "answer": parsed_answer,
            "correct": correct,
            "actual_answer": ds[q_id]["answer"]
        })
    result_obj = {
        "question_id": q_id,
        "question": question,
        "resample_results": resample_results
    }
    with open("../resample_results/options_results.jsonl", "a") as f:
        f.write(json.dumps(result_obj) + "\n")
    print(f"Resampled question {q_id}")        

for q_id in range(100,150):
    for line in read_jsonl("../baseline/baseline_CoTs_options.jsonl"):
        if line["question_id"] == q_id:
            if line["complete_reason"] == "stop":
                print(f"Resampling question {q_id}...")
                resample(q_id)
            else:
                print(f"Skipping question {q_id} because baseline CoT was not complete.")
            break
