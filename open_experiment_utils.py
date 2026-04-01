from __future__ import annotations

import re
from pathlib import Path

from options_experiment_utils import (
    FULL_TRACE_MAX_COMPLETION_TOKENS,
    FULL_TRACE_TEMPERATURE,
    FULL_TRACE_TOP_P,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    MODEL_ID,
    RANDOM_CONTROL_DEFAULT_SEED,
    SUPPORTED_RESAMPLE_CONDITIONS,
    build_resample_condition_full_token_ids,
    decode_token_ids,
    decile_prefix_token_count,
    extract_final_answer_text,
    extract_reasoning_trace,
    get_reasoning_token_count,
    get_reasoning_token_ids,
    read_jsonl,
    validate_resample_condition,
    write_jsonl,
)


OPEN_BASELINE_SCHEMA_VERSION = 6
OPEN_PROBE_METHOD_VERSION = 1
OPEN_RESAMPLE_SCHEMA_VERSION = 6
OPEN_SCORING_MODE = "gold_answer_correctness"
OPEN_JUDGE_MODEL_ID = "openai/gpt-oss-20b"
OPEN_DEFAULT_DATASET_CATEGORY = "math"
OPEN_VERDICT_CORRECT = "CORRECT"
OPEN_VERDICT_INCORRECT = "INCORRECT"
OPEN_VERDICT_UNKNOWN = "UNKNOWN"

OPEN_SYSTEM_PROMPT = (
    "Solve the following problem. You may reason step by step. If you produce "
    "a reasoning trace, keep it inside <think>...</think>. After the reasoning, "
    "return a concise final answer in plain text. Do not use multiple-choice "
    "option letters."
)

OPEN_PROBE_SYSTEM_PROMPT = (
    "You are answering a question using a partial reasoning trace that was "
    "already produced for the same question. Do not continue the reasoning "
    "trace. Do not start a new chain of thought. Do not output <think> or "
    "</think>. Return only a concise final answer in plain text. Do not use "
    "multiple-choice option letters."
)

OPEN_PROBE_RETRY_SYSTEM_PROMPT = (
    OPEN_PROBE_SYSTEM_PROMPT
    + " If you include explanation or more than one short sentence, the answer "
    "will be discarded."
)

OPEN_PROBE_MAX_COMPLETION_TOKENS = 96
OPEN_ANSWER_MARKER_PATTERNS = (
    r"final answer",
    r"answer",
    r"short answer",
    r"final response",
    r"response",
)

ROOT_DIR = Path(__file__).resolve().parent
BASELINE_OPEN_PATH = ROOT_DIR / "baseline" / "baseline_CoTs_open.jsonl"
OPEN_RESULTS_PATH = ROOT_DIR / "resample_results" / "open_results.jsonl"
RANDOM_OPEN_RESULTS_PATH = ROOT_DIR / "resample_results" / "open_random_results.jsonl"
SHUFFLE_OPEN_RESULTS_PATH = ROOT_DIR / "resample_results" / "open_shuffle_results.jsonl"


def get_open_results_path(condition):
    condition = validate_resample_condition(condition)
    if condition == "original":
        return OPEN_RESULTS_PATH
    if condition == "random":
        return RANDOM_OPEN_RESULTS_PATH
    if condition == "shuffle":
        return SHUFFLE_OPEN_RESULTS_PATH
    raise AssertionError(f"Unhandled condition: {condition}")


def build_open_prompt(question):
    return question


def build_open_probe_messages(prompt, reasoning_prefix, retry=False):
    reasoning_excerpt = (reasoning_prefix or "").strip() or "[empty partial reasoning trace]"
    system_prompt = OPEN_PROBE_RETRY_SYSTEM_PROMPT if retry else OPEN_PROBE_SYSTEM_PROMPT
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
                "and the partial trace above, return only the final answer in "
                "plain text.\n\n"
                "Final answer:"
            ),
        },
    ]


def probe_open_response_needs_retry(raw_probe_response, response):
    candidate = (response or raw_probe_response or "").strip()
    if not candidate:
        return True

    lowered_raw = (raw_probe_response or "").lower()
    if "<think>" in lowered_raw or "</think>" in lowered_raw:
        return True

    reasoning_markers = (
        "let's",
        "step by step",
        "reasoning",
        "analy",
        "first,",
        "therefore,",
        "because",
    )
    if any(marker in lowered_raw for marker in reasoning_markers) and len(candidate.split()) > 12:
        return True

    if len(candidate.split()) > 48:
        return True

    sentence_like_terminators = candidate.count(".") + candidate.count("!") + candidate.count("?")
    return sentence_like_terminators > 1


def probe_open_answer(client, prompt, reasoning_prefix):
    attempts = (False, True)
    last_raw_probe_response = ""
    last_response = ""

    for retry in attempts:
        completion = client.chat.completions.create(
            messages=build_open_probe_messages(prompt, reasoning_prefix, retry=retry),
            model=MODEL_ID,
            temperature=0.0,
            top_p=1.0,
            max_completion_tokens=OPEN_PROBE_MAX_COMPLETION_TOKENS,
        )
        raw_probe_response = completion.choices[0].message.content or ""
        response = extract_final_answer_text(raw_probe_response)

        last_raw_probe_response = raw_probe_response
        last_response = response
        if not probe_open_response_needs_retry(raw_probe_response, response):
            break

    return last_raw_probe_response, last_response


def _strip_answer_scaffolding(text):
    candidate = (text or "").strip()
    candidate = candidate.strip("`*_")
    prefixes = (
        r"^(final answer|answer|short answer)\s*[:\-]\s*",
        r"^(the answer is|it is|it's|therefore|thus)\s+",
    )
    changed = True
    while changed and candidate:
        changed = False
        for pattern in prefixes:
            updated = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
            if updated != candidate:
                candidate = updated
                changed = True
    return candidate.strip(" \t\r\n\"'`()[]{}.,;:!?")


def _extract_marked_answer_spans(text):
    patterns = (
        r"\*\*(.+?)\*\*",
        r"__(.+?)__",
        r"`([^`\n]+)`",
    )
    spans = []
    seen = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text or "", flags=re.DOTALL):
            span = _strip_answer_scaffolding(match.group(1))
            if not span or len(span.split()) > 8:
                continue
            key = span.casefold()
            if key in seen:
                continue
            seen.add(key)
            spans.append(span)

    return spans


def _looks_like_answer_candidate(text):
    candidate = (text or "").strip()
    if not candidate:
        return False

    lowered = candidate.casefold()
    if lowered in {"unknown", "n/a", "none"}:
        return False

    reasoning_markers = (
        "let's",
        "step by step",
        "because",
        "therefore",
        "this means",
        "i think",
        "we need to",
        "the question is",
    )
    if any(marker in lowered for marker in reasoning_markers):
        return False

    if len(candidate.split()) > 24:
        return False

    return True


def _token_set(text):
    return set(_normalise_free_form_text(text).split())


def _has_high_token_overlap(left, right, threshold=0.7):
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return overlap >= threshold


def _extract_after_last_answer_marker(text):
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:\*\*|__|`)?\s*(?:"
            + "|".join(OPEN_ANSWER_MARKER_PATTERNS)
            + r")\s*(?:\*\*|__|`)?\s*[:\-]?\s*",
            text or "",
        )
    )
    if not matches:
        return ""

    candidate = (text or "")[matches[-1].end() :].strip()
    return candidate


def _strip_question_echo(candidate, question):
    cleaned_candidate = _strip_answer_scaffolding(candidate)
    if not cleaned_candidate or not question:
        return cleaned_candidate

    if "___" in question and ":" in cleaned_candidate and ":" in question:
        question_head = question.split(":", 1)[0]
        candidate_head, candidate_tail = cleaned_candidate.split(":", 1)
        if _has_high_token_overlap(candidate_head, question_head):
            stripped_tail = _strip_answer_scaffolding(candidate_tail)
            if stripped_tail:
                return stripped_tail

    return cleaned_candidate


def _extract_boxed_answer(text):
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text or "")
    if not matches:
        return ""

    candidate = _strip_answer_scaffolding(matches[-1])
    return candidate


def _extract_last_answer_like_line(text, question=None):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        boxed_candidate = _extract_boxed_answer(line)
        if boxed_candidate:
            return boxed_candidate
        marked_spans = _extract_marked_answer_spans(line)
        if marked_spans and len(" ".join(marked_spans).split()) <= 12:
            return ", ".join(marked_spans)
        cleaned_line = _strip_question_echo(line, question)
        if _looks_like_answer_candidate(cleaned_line):
            return cleaned_line
    return ""


def extract_open_answer_candidate(text, question=None):
    full_text = (text or "").strip()
    if not full_text:
        return ""

    marker_candidate = _extract_after_last_answer_marker(full_text)
    if marker_candidate:
        boxed_marker_candidate = _extract_boxed_answer(marker_candidate)
        if boxed_marker_candidate:
            return boxed_marker_candidate
        stripped_marker_candidate = _strip_question_echo(marker_candidate, question)
        answer_like_line = _extract_last_answer_like_line(stripped_marker_candidate, question)
        if answer_like_line:
            return answer_like_line
        marked_spans = _extract_marked_answer_spans(stripped_marker_candidate)
        if marked_spans and len(" ".join(marked_spans).split()) <= 12:
            return ", ".join(marked_spans)
        cleaned_marker_candidate = _strip_answer_scaffolding(stripped_marker_candidate)
        if cleaned_marker_candidate:
            return cleaned_marker_candidate

    tail = extract_final_answer_text(full_text).strip()
    if not tail:
        tail = full_text

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", tail) if part.strip()]
    for paragraph in reversed(paragraphs):
        boxed_candidate = _extract_boxed_answer(paragraph)
        if boxed_candidate:
            return boxed_candidate
        marked_spans = _extract_marked_answer_spans(paragraph)
        if marked_spans and len(" ".join(marked_spans).split()) <= 12:
            return ", ".join(marked_spans)
        cleaned_paragraph = _strip_question_echo(paragraph, question)
        if _looks_like_answer_candidate(cleaned_paragraph):
            return cleaned_paragraph

    answer_like_line = _extract_last_answer_like_line(tail, question)
    if answer_like_line:
        return answer_like_line

    stripped_tail = _strip_question_echo(tail, question)
    boxed_tail = _extract_boxed_answer(stripped_tail)
    if boxed_tail:
        return boxed_tail
    marked_spans = _extract_marked_answer_spans(stripped_tail)
    if marked_spans and len(" ".join(marked_spans).split()) <= 12:
        return ", ".join(marked_spans)
    return stripped_tail


def _normalise_free_form_text(text):
    candidate = _strip_answer_scaffolding(text)
    candidate = candidate.casefold()
    candidate = re.sub(r"[^0-9a-z]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def _normalise_without_articles(text):
    candidate = _normalise_free_form_text(text)
    candidate = re.sub(r"\b(a|an|the)\b", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def _split_answer_fragments(text):
    fragments = []
    seen = set()
    for piece in re.split(r"\s*,\s*|\s*;\s*", text or ""):
        fragment = _normalise_without_articles(piece) or _normalise_free_form_text(piece)
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)
        fragments.append(fragment)
    return fragments


def _is_explicit_unknown_answer(text):
    normalised = _normalise_without_articles(text)
    return normalised in {
        "",
        "unknown",
        "none",
        "n a",
        "not sure",
        "cannot determine",
        "cant determine",
    }


def direct_score_open_answer_correctness(response, gold_answer):
    if _is_explicit_unknown_answer(response):
        return OPEN_VERDICT_INCORRECT, "direct_empty_or_unknown"

    normalised_response = _normalise_free_form_text(response)
    article_stripped_response = _normalise_without_articles(response)
    normalised_gold = _normalise_free_form_text(gold_answer)
    article_stripped_gold = _normalise_without_articles(gold_answer)

    if (
        normalised_response
        and normalised_gold
        and normalised_response == normalised_gold
    ) or (
        article_stripped_response
        and article_stripped_gold
        and article_stripped_response == article_stripped_gold
    ):
        return OPEN_VERDICT_CORRECT, "direct_exact_match"

    response_fragments = _split_answer_fragments(response)
    gold_fragments = _split_answer_fragments(gold_answer)
    if (
        len(response_fragments) > 1
        and len(gold_fragments) > 1
        and set(response_fragments) == set(gold_fragments)
    ):
        return OPEN_VERDICT_CORRECT, "direct_unordered_fragment_match"

    return None, None


def _extract_text_from_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "output_text"}:
                text = item.get("text") or item.get("output_text") or ""
                if text:
                    chunks.append(text)
        return "".join(chunks).strip()
    return str(content).strip()


def _extract_completion_text(choice):
    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content_text = _extract_text_from_content(getattr(message, "content", None))
    if content_text:
        return content_text

    for attr_name in ("reasoning", "reasoning_content"):
        reasoning_text = _extract_text_from_content(getattr(message, attr_name, None))
        if reasoning_text:
            return reasoning_text

    return ""


def _extract_judge_verdict(judge_response):
    if not judge_response:
        return None

    tagged_matches = re.findall(
        rf"<verdict>\s*({OPEN_VERDICT_CORRECT}|{OPEN_VERDICT_INCORRECT})\s*</verdict>",
        judge_response,
        flags=re.IGNORECASE,
    )
    if tagged_matches:
        return tagged_matches[-1].upper()

    line_matches = re.findall(
        rf"(?im)^\s*(?:verdict\s*:\s*)?({OPEN_VERDICT_CORRECT}|{OPEN_VERDICT_INCORRECT})\s*$",
        judge_response,
        flags=re.IGNORECASE,
    )
    if line_matches:
        return line_matches[-1].upper()

    stripped = judge_response.strip()
    exact_match = re.fullmatch(
        rf"({OPEN_VERDICT_CORRECT}|{OPEN_VERDICT_INCORRECT})",
        stripped,
        flags=re.IGNORECASE,
    )
    if exact_match:
        return exact_match.group(1).upper()

    return None


def _judge_open_answer_correctness_once(
    client,
    question,
    final_answer_text,
    gold_answer,
    retry=False,
):
    system_prompt = (
        "You are a strict grader. Decide whether the model's free-form answer "
        "should count as correct for the given question when compared with the "
        "gold answer. Accept paraphrases, synonyms, and standard equivalent "
        "formulations that preserve meaning. Treat logically, mathematically, "
        "or definitionally equivalent answers as CORRECT even when they use a "
        "different but standard characterization, criterion, or theorem form "
        "than the gold answer. In particular, if the gold answer gives one "
        "necessary-and-sufficient condition, and the model answer gives another "
        "standard necessary-and-sufficient condition for the same concept in "
        "the same context, count it as CORRECT. Do not require the model answer "
        "to match the same minimal wording or compressed form as the gold "
        "answer. Do not mark an answer INCORRECT merely because it is longer, "
        "more explicit, or includes true supporting explanation. Only mark "
        "INCORRECT if the answer is missing essential content, adds material "
        "that makes it incompatible with the gold answer, or states a genuinely "
        "different condition or conclusion. Do not solve the question yourself. "
        "Keep your reasoning succinct and focused on the comparison, and end with "
        "exactly one final line containing "
        "<verdict>CORRECT</verdict> or <verdict>INCORRECT</verdict>."
    )
    if retry:
        system_prompt += " Keep the reasoning concise and preserve the same final verdict tag format."

    user_sections = [
        f"Question: {question}",
        "Model final answer text:",
        final_answer_text or "[empty]",
        f"Gold answer: {gold_answer}",
        "Verdict format:",
        "<verdict>CORRECT</verdict>",
        "or",
        "<verdict>INCORRECT</verdict>",
    ]
    user_sections.extend(
        [
            "Evaluate the answer and end with the required verdict tag line."
        ]
    )

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "\n\n".join(user_sections),
            },
        ],
        model=OPEN_JUDGE_MODEL_ID,
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
    )
    choice = completion.choices[0]
    judge_response = _extract_completion_text(choice)
    judge_finish_reason = getattr(choice, "finish_reason", None)
    return judge_response, _extract_judge_verdict(judge_response), judge_finish_reason


def resolve_open_answer_correctness(client, question, response, gold_answer, raw_response=None):
    final_answer_text = extract_final_answer_text(raw_response or response).strip()
    if not final_answer_text:
        final_answer_text = (response or "").strip()

    answer_candidate = extract_open_answer_candidate(final_answer_text, question)
    direct_verdict, direct_source = direct_score_open_answer_correctness(
        answer_candidate,
        gold_answer,
    )
    if answer_candidate != final_answer_text:
        response_direct_verdict, response_direct_source = direct_score_open_answer_correctness(
            final_answer_text,
            gold_answer,
        )
        if response_direct_verdict == OPEN_VERDICT_CORRECT:
            direct_verdict = response_direct_verdict
            direct_source = f"{response_direct_source}_full_response"
        elif direct_verdict is None and response_direct_verdict == OPEN_VERDICT_INCORRECT:
            direct_verdict = response_direct_verdict
            direct_source = f"{response_direct_source}_full_response"

    if direct_verdict is not None:
        return {
            "judge_response": None,
            "judge_finish_reason": None,
            "judge_attempt_count": 0,
            "answer_candidate": answer_candidate,
            "direct_verdict": direct_verdict,
            "judge_verdict": None,
            "verdict": direct_verdict,
            "correct": direct_verdict == OPEN_VERDICT_CORRECT,
            "judge_parsed_answer": None,
            "used_llm_judge": False,
            "verdict_source": direct_source,
        }

    last_judge_response = None
    first_nonempty_judge_response = None
    last_finish_reason = None
    attempt_count = 0
    for retry in (False, True):
        judge_response, judge_verdict, judge_finish_reason = _judge_open_answer_correctness_once(
            client,
            question,
            final_answer_text,
            gold_answer,
            retry=retry,
        )
        attempt_count += 1
        last_finish_reason = judge_finish_reason
        if judge_response and first_nonempty_judge_response is None:
            first_nonempty_judge_response = judge_response
        if judge_response:
            last_judge_response = judge_response
        if judge_verdict:
            return {
                "judge_response": judge_response or first_nonempty_judge_response,
                "judge_finish_reason": judge_finish_reason,
                "judge_attempt_count": attempt_count,
                "answer_candidate": answer_candidate,
                "direct_verdict": None,
                "judge_verdict": judge_verdict,
                "verdict": judge_verdict,
                "correct": judge_verdict == OPEN_VERDICT_CORRECT,
                "judge_parsed_answer": judge_verdict,
                "used_llm_judge": True,
                "verdict_source": "llm_judge",
            }

    return {
        "judge_response": last_judge_response or first_nonempty_judge_response or "",
        "judge_finish_reason": last_finish_reason,
        "judge_attempt_count": attempt_count,
        "answer_candidate": answer_candidate,
        "direct_verdict": None,
        "judge_verdict": OPEN_VERDICT_UNKNOWN,
        "verdict": OPEN_VERDICT_UNKNOWN,
        "correct": False,
        "judge_parsed_answer": OPEN_VERDICT_UNKNOWN,
        "used_llm_judge": True,
        "verdict_source": "unknown",
    }
