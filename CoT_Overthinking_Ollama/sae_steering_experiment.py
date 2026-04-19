"""Steering experiment: does adding/removing SAE feature 4103 at layer 17
change whether the model gets the answer right?

We add `alpha * unit(W_dec[feature_id])` to the residual stream at layer 17
via a forward hook on `model.model.language_model.layers[17]`, during both
prompt processing and greedy generation, and compare the final letter answer
against the alpha=0 baseline for the same question.

Question sets (drawn from the existing ollama resample run, so "firing / not
firing" refers to feature activation under the HF forward pass at the last
prompt token — we re-derive the sets here to avoid any stale cache):

- A (suppression target): right->wrong questions where feature 4103 fires.
    Hypothesis: alpha < 0 should flip some to correct.
- B (selectivity control): right->right questions where feature 4103 fires.
    Hypothesis: alpha < 0 should not badly hurt these.
- C (induction): right->right questions where feature 4103 is silent.
    Hypothesis: alpha > 0 should flip some to wrong.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import (  # noqa: E402
    read_jsonl, write_jsonl,
    extract_answer_label, extract_final_answer_text,
)
from ollama_options import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT  # noqa: E402

MODEL_ID = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_16k_l0_medium"
LAYER = 17
FEATURE_ID = 4103

PER_Q_PATH = SCRIPT_DIR / "resample_results" / "sae_l17_w16k_l0medium_per_question.jsonl"
RESAMPLE_PATH = SCRIPT_DIR / "resample_results" / "options_results_ollama.jsonl"
BASELINE_PATH = SCRIPT_DIR / "baseline" / "baseline_CoTs_options_ollama.jsonl"

OUT_DIR = SCRIPT_DIR / "resample_results"
DEFAULT_OUTPUT = OUT_DIR / f"steering_feature_{FEATURE_ID}.jsonl"
DEFAULT_REPORT = OUT_DIR / f"steering_feature_{FEATURE_ID}.md"

DEFAULT_ALPHAS = [-500.0, -200.0, 0.0, 200.0, 500.0]
DEFAULT_MAX_NEW_TOKENS = 3072


def build_prompt(tokenizer, user_text):
    text = (
        f"<start_of_turn>user\n{BASELINE_SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=True)


def partition_questions(per_q_path, resample_path, baseline_path, feature_id):
    per_q = {r["question_id"]: dict(r["active_features"]) for r in read_jsonl(per_q_path)}
    classes = {r["question_id"]: r["class"] for r in read_jsonl(per_q_path)}
    q_meta = {row["question_id"]: row for row in read_jsonl(resample_path)}
    baseline_rows = {row["question_id"]: row for row in read_jsonl(baseline_path)}

    A, B, C = [], [], []
    for qid, cls in classes.items():
        active = per_q[qid]
        meta = q_meta.get(qid)
        base = baseline_rows.get(qid)
        if meta is None or base is None:
            continue
        entry = {
            "question_id": qid,
            "prompt": meta["prompt"],
            "question": meta["question"],
            "actual_answer_label": base["actual_answer_label"],
            "feature_activation": float(active.get(feature_id, 0.0)),
            "ollama_class": cls,
        }
        fires = entry["feature_activation"] > 0
        if cls == "right_wrong" and fires:
            A.append(entry)
        elif cls == "right_right" and fires:
            B.append(entry)
        elif cls == "right_right" and not fires:
            C.append(entry)
    A.sort(key=lambda e: e["feature_activation"], reverse=True)
    B.sort(key=lambda e: e["feature_activation"], reverse=True)
    return A, B, C


def make_steering_hook(alpha, direction, mode="add"):
    """Forward hook at a decoder layer's output.

    mode='add':    residual += alpha * direction (boost/suppress additively)
    mode='ablate': residual -= (residual · direction) * direction  (project out)
                   alpha is ignored; this zero-ablates the direction.
    """
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        d = direction.to(hs.dtype)
        if mode == "ablate":
            proj = (hs * d).sum(dim=-1, keepdim=True)
            hs = hs - proj * d
        else:
            hs = hs + alpha * d
        return (hs,) + output[1:] if is_tuple else hs
    return hook


def generate_answer(model, tokenizer, prompt_inputs, max_new_tokens):
    with torch.inference_mode():
        out = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                        help="Steering magnitudes to sweep (unit-normed direction).")
    parser.add_argument("--include-ablate", action="store_true",
                        help="Also run an 'ablate' condition that projects the direction out entirely.")
    parser.add_argument("--n-per-set", type=int, default=12,
                        help="Max questions per set A/B/C.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--feature-id", type=int, default=FEATURE_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print set sizes and exit without running.")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Partitioning questions on feature {args.feature_id}...")
    A, B, C = partition_questions(PER_Q_PATH, RESAMPLE_PATH, BASELINE_PATH, args.feature_id)
    rng.shuffle(C)
    A = A[: args.n_per_set]
    B = B[: args.n_per_set]
    C = C[: args.n_per_set]
    for q in A: q["set"] = "A"
    for q in B: q["set"] = "B"
    for q in C: q["set"] = "C"
    all_questions = A + B + C
    print(f"  A (RW, 4103 fires)    = {len(A)}")
    print(f"  B (RR, 4103 fires)    = {len(B)}")
    print(f"  C (RR, 4103 silent)   = {len(C)}")
    print(f"  total questions       = {len(all_questions)}")
    print(f"  alphas                = {args.alphas}")
    print(f"  total generations     = {len(all_questions) * len(args.alphas)}")
    if args.dry_run:
        return

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    print(f"Loading SAE {SAE_RELEASE}/{SAE_ID}...")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    W_dec = sae.W_dec.detach()  # (d_sae, d_in)
    raw_dir = W_dec[args.feature_id].float()
    unit_dir = (raw_dir / raw_dir.norm()).to("cuda")
    print(f"  W_dec[{args.feature_id}] norm = {float(raw_dir.norm().item()):.4f}")

    layer_module = model.model.language_model.layers[LAYER]

    conditions = [("add", a) for a in args.alphas]
    if args.include_ablate:
        conditions.append(("ablate", 0.0))

    results = []
    n_total = len(all_questions) * len(conditions)
    step = 0
    for q in all_questions:
        prompt_inputs = build_prompt(tokenizer, q["prompt"])
        prompt_inputs = {k: v.to("cuda") for k, v in prompt_inputs.items()}
        for mode, alpha in conditions:
            step += 1
            if mode == "add" and alpha == 0.0:
                handle = None
            else:
                handle = layer_module.register_forward_hook(
                    make_steering_hook(alpha, unit_dir, mode=mode),
                )
            try:
                gen_text = generate_answer(
                    model, tokenizer, prompt_inputs, args.max_new_tokens,
                )
            finally:
                if handle is not None:
                    handle.remove()

            final_text = extract_final_answer_text(gen_text)
            parsed = extract_answer_label(final_text)
            actual = q["actual_answer_label"]
            correct = parsed == actual
            cond_label = "ablate" if mode == "ablate" else f"{alpha:+.0f}"
            results.append({
                "question_id": q["question_id"],
                "set": q["set"],
                "ollama_class": q["ollama_class"],
                "feature_activation_at_prompt": q["feature_activation"],
                "mode": mode,
                "alpha": alpha,
                "condition": cond_label,
                "actual_answer_label": actual,
                "parsed_answer": parsed,
                "correct": bool(correct),
                "final_answer_text": final_text[:400],
                "generation_chars": len(gen_text),
            })
            print(f"  [{step}/{n_total}] qid={q['question_id']} set={q['set']} "
                  f"cond={cond_label} parsed={parsed} (actual={actual}) "
                  f"correct={correct}")

    write_jsonl(args.output_path, results)
    print(f"\nWrote {len(results)} rows to {args.output_path}")
    write_report(args, results, A, B, C)


def write_report(args, results, A, B, C):
    # Preserve the order in which conditions appear (matches what main ran)
    conditions = []
    seen = set()
    for r in results:
        c = r["condition"]
        if c not in seen:
            seen.add(c)
            conditions.append(c)
    baseline_cond = "+0"  # the α=0 additive is the baseline

    pivot = {(r["question_id"], r["condition"]): r for r in results}

    def summarise(question_list, set_name):
        lines = [f"### Set {set_name}: {len(question_list)} questions", ""]
        lines.append("| qid | actual | " + " | ".join(conditions) + " |")
        lines.append("|---" * (len(conditions) + 2) + "|")
        for q in question_list:
            qid = q["question_id"]
            cells = []
            for c in conditions:
                r = pivot.get((qid, c))
                if r is None:
                    cells.append("—")
                else:
                    mark = "✓" if r["correct"] else "✗"
                    cells.append(f"{r['parsed_answer']}{mark}")
            lines.append(f"| {qid} | {q['actual_answer_label']} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append(f"**Flips vs {baseline_cond} baseline (within-experiment):**")
        lines.append("")
        lines.append("| condition | same | correct→wrong | wrong→correct | wrong→different-wrong |")
        lines.append("|---|---|---|---|---|")
        for c in conditions:
            if c == baseline_cond:
                lines.append(f"| {c} | (baseline) | - | - | - |")
                continue
            same = c2w = w2c = w2ow = 0
            for q in question_list:
                qid = q["question_id"]
                b = pivot.get((qid, baseline_cond))
                r = pivot.get((qid, c))
                if b is None or r is None:
                    continue
                if b["parsed_answer"] == r["parsed_answer"]:
                    same += 1
                elif b["correct"] and not r["correct"]:
                    c2w += 1
                elif (not b["correct"]) and r["correct"]:
                    w2c += 1
                else:
                    w2ow += 1
            lines.append(f"| {c} | {same} | {c2w} | {w2c} | {w2ow} |")
        lines.append("")
        lines.append("**Accuracy per condition:**")
        lines.append("")
        lines.append("| condition | correct / n | % |")
        lines.append("|---|---|---|")
        for c in conditions:
            n = c_count = 0
            for q in question_list:
                r = pivot.get((q["question_id"], c))
                if r is None:
                    continue
                n += 1
                if r["correct"]:
                    c_count += 1
            pct = (100 * c_count / n) if n else 0.0
            lines.append(f"| {c} | {c_count}/{n} | {pct:.0f} |")
        lines.append("")
        return lines

    out = []
    out.append(f"# Steering experiment: feature {args.feature_id} at layer {LAYER}")
    out.append("")
    out.append(f"Model `{MODEL_ID}`, SAE `{SAE_RELEASE}/{SAE_ID}`, greedy decode, "
               f"max_new_tokens={args.max_new_tokens}.")
    out.append("")
    out.append("Intervention: residual stream at layer 17 output += α · unit(W_dec[4103]) "
               "at every token during prompt + generation.")
    out.append("")
    out.append("Legend: `X✓` = parsed answer X, matches actual; `X✗` = parsed X, wrong; "
               "`None✗` = failed to parse a letter.")
    out.append("")
    out.append("## Set A — right->wrong (ollama) + feature fires at prompt")
    out.append("*Hypothesis: suppression (α < 0) should flip some wrong → correct.*")
    out.append("")
    out.extend(summarise(A, "A"))
    out.append("## Set B — right->right + feature fires (selectivity control)")
    out.append("*If suppression helps selectively on A, it should not ruin B.*")
    out.append("")
    out.extend(summarise(B, "B"))
    out.append("## Set C — right->right + feature silent")
    out.append("*Hypothesis: induction (α > 0) should flip some correct → wrong.*")
    out.append("")
    out.extend(summarise(C, "C"))
    args.report_path.write_text("\n".join(out) + "\n")
    print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
