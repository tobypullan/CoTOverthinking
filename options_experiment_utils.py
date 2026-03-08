from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path


MODEL_ID = "qwen/qwen3-32b"
TOKENIZER_MODEL_ID = "Qwen/Qwen3-32B"
JUDGE_MODEL_ID = "llama-3.1-8b-instant"
PROBE_METHOD_VERSION = 2

SYSTEM_PROMPT = (
    "Solve the following problem. Please make sure that your response only "
    "consists of a single letter corresponding to the correct answer choice. "
    "Do not include anything else in your final response."
)

PROBE_SYSTEM_PROMPT = (
    "You are answering a multiple-choice question with answer choices labelled "
    "A through J. You will receive the question and a partial reasoning trace "
    "that was already produced for the same question. Do not continue the "
    "reasoning trace. Do not start a new chain of thought. Do not output "
    "<think> or </think>. Return exactly one uppercase answer letter."
)

PROBE_RETRY_SYSTEM_PROMPT = (
    PROBE_SYSTEM_PROMPT
    + " If you output anything other than a single letter, the answer will be "
    "discarded."
)

FULL_TRACE_TEMPERATURE = 0.6
FULL_TRACE_TOP_P = 0.95
FULL_TRACE_MAX_COMPLETION_TOKENS = 38912

PROBE_TEMPERATURE = 0.0
PROBE_TOP_P = 1.0
PROBE_MAX_COMPLETION_TOKENS = 8

JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_MAX_COMPLETION_TOKENS = 8

LABELS = tuple("ABCDEFGHIJ")
LABEL_SET = set(LABELS)

ROOT_DIR = Path(__file__).resolve().parent
BASELINE_OPTIONS_PATH = ROOT_DIR / "baseline" / "baseline_CoTs_options.jsonl"
OPTIONS_RESULTS_PATH = ROOT_DIR / "resample_results" / "options_results.jsonl"


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def build_options_prompt(question, options):
    labelled_options = [f"\n{LABELS[i]}: {options[i]}" for i in range(len(options))]
    return question + " The options are: " + "".join(labelled_options)


def extract_options_from_prompt(prompt):
    parts = prompt.split(" The options are: ", 1)
    if len(parts) != 2:
        raise ValueError("Prompt does not contain a parsable options section.")

    option_lines = [line for line in parts[1].splitlines() if line.strip()]
    options = []
    for line in option_lines:
        if ": " not in line:
            raise ValueError(f"Malformed option line: {line}")
        options.append(line.split(": ", 1)[1])
    return options


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
    if stripped in LABEL_SET:
        return stripped

    return None


def extract_reasoning_trace(response):
    if not response:
        return None

    closed_match = re.search(r"<think>\s*(.*?)\s*</think>", response, flags=re.DOTALL)
    if closed_match:
        return closed_match.group(1).strip()

    start = response.find("<think>")
    if start == -1:
        return None

    return response[start + len("<think>") :].strip()


def extract_final_answer_text(response):
    if not response:
        return ""

    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()

    return response.strip()


def judge_answer_label(client, response, options):
    labelled_options = [f"{LABELS[i]}: {options[i]}" for i in range(len(options))]
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract the answer label already chosen in another "
                    "model's response. Do not solve the question yourself. You "
                    "may use the provided options only to map quoted or "
                    "paraphrased option text back to a label. If the response "
                    "does not clearly choose exactly one option, return only "
                    "UNKNOWN."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Options: {labelled_options}\n\n"
                    f"Model response: {response}\n\n"
                    "Return only one token: A, B, C, D, E, F, G, H, I, J, or "
                    "UNKNOWN."
                ),
            },
        ],
        model=JUDGE_MODEL_ID,
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
    )
    judge_response = completion.choices[0].message.content.strip()
    judged_label = extract_answer_label(judge_response)
    if judged_label:
        return judge_response, judged_label

    direct_label = extract_answer_label(response)
    if direct_label:
        return judge_response, direct_label

    return judge_response, "UNKNOWN"


@lru_cache(maxsize=1)
def get_qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Token-based deciles require transformers>=4.51.0. Install "
            "transformers before running the options experiment."
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer '{TOKENIZER_MODEL_ID}'."
        ) from exc


def get_reasoning_token_ids(reasoning_trace):
    tokenizer = get_qwen_tokenizer()
    return tokenizer.encode(reasoning_trace or "", add_special_tokens=False)


def get_reasoning_token_count(reasoning_trace):
    return len(get_reasoning_token_ids(reasoning_trace))


def decile_prefix_token_count(total_tokens, point):
    if point <= 0:
        return 0
    if point >= 1:
        return total_tokens
    return math.ceil(total_tokens * point)


def truncate_reasoning_trace(reasoning_trace, token_count):
    if token_count <= 0:
        return ""

    tokenizer = get_qwen_tokenizer()
    token_ids = get_reasoning_token_ids(reasoning_trace)
    prefix_ids = token_ids[:token_count]
    return tokenizer.decode(
        prefix_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_reasoning_excerpt(reasoning_prefix):
    prefix = (reasoning_prefix or "").strip()
    if not prefix:
        return "[empty partial reasoning trace]"
    return prefix


def build_probe_messages(prompt, reasoning_prefix, retry=False):
    reasoning_excerpt = build_reasoning_excerpt(reasoning_prefix)
    system_prompt = PROBE_RETRY_SYSTEM_PROMPT if retry else PROBE_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "/no_think\n"
                f"Question:\n{prompt}\n\n"
                "Partial reasoning trace already produced for this same "
                "question:\n"
                "```text\n"
                f"{reasoning_excerpt}\n"
                "```\n\n"
                "Do not continue or rewrite that trace. Based on the question "
                "and the partial trace above, return exactly one uppercase "
                "answer letter from A to J.\n\n"
                "Answer:"
            ),
        },
    ]


def probe_response_needs_retry(raw_probe_response, response):
    candidate = (response or raw_probe_response or "").strip()
    if not candidate:
        return True

    lowered_raw = (raw_probe_response or "").lower()
    if "<think>" in lowered_raw or "</think>" in lowered_raw:
        return True

    if extract_answer_label(candidate):
        return False

    reasoning_markers = (
        "let's",
        "step by step",
        "reasoning",
        "analy",
        "tackle this question",
    )
    if any(marker in lowered_raw for marker in reasoning_markers):
        return True

    return len(candidate.split()) > 6
