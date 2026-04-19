"""Validate UNKNOWN parsed answers by inspecting next-token letter logits.

For each sampled UNKNOWN probe, we reconstruct the exact probe prompt that
Ollama saw (matching its gemma3 chat template), run a forward pass through
gemma-3-4b-it locally, and measure the probability distribution over the
ten answer-letter tokens A..J at the next-token position. If the model's
letter-restricted argmax disagrees with the correct answer, or the letter
probability mass is small (model not in commit mode), UNKNOWN is justified.
If the argmax is the correct answer with meaningful mass, the UNKNOWN was
lost data.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import read_jsonl, write_jsonl  # noqa: E402
from ollama_options import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT  # noqa: E402


MODEL_ID = "google/gemma-3-4b-it"
LABELS = tuple("ABCDEFGHIJ")
LOST_DATA_MASS_THRESHOLD = 0.1

RESAMPLE_FILES = {
    "original": SCRIPT_DIR / "resample_results" / "options_results_ollama.jsonl",
    "shuffle": SCRIPT_DIR / "resample_results" / "options_shuffle_results_ollama.jsonl",
    "random": SCRIPT_DIR / "resample_results" / "options_random_results_ollama.jsonl",
}
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "resample_results" / "unknown_logit_validation.jsonl"


def build_ollama_gemma_prompt(system_text, user_text, assistant_prefill):
    """Manually mirror ollama's gemma3 chat template for [system, user, assistant]."""
    return (
        f"<start_of_turn>user\n{system_text}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant_prefill}"
    )


def find_letter_token_ids(tokenizer):
    ids_by_letter = {}
    for letter in LABELS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"Letter {letter!r} did not tokenize to a single token: {encoded}"
            )
        ids_by_letter[letter] = encoded[0]
    return ids_by_letter


def collect_unknown_probes(conditions, min_decile):
    unknowns = []
    for condition in conditions:
        path = RESAMPLE_FILES[condition]
        for row in read_jsonl(path):
            q_id = row["question_id"]
            prompt = row["prompt"]
            for p in row["resample_results"]:
                if p["parsed_answer"] != "UNKNOWN":
                    continue
                if p["resample_point"] < min_decile:
                    continue
                unknowns.append({
                    "condition": condition,
                    "question_id": q_id,
                    "decile": p["resample_point"],
                    "prompt": prompt,
                    "reasoning_prefix": p["injected_prefix_text"],
                    "actual_answer_label": p["actual_answer_label"],
                    "raw_probe_response": p["raw_probe_response"],
                })
    return unknowns


def stratified_sample(unknowns, sample_size_per_condition, rng):
    by_cond = {}
    for u in unknowns:
        by_cond.setdefault(u["condition"], []).append(u)
    sampled = []
    for cond, lst in by_cond.items():
        rng.shuffle(lst)
        sampled.extend(lst[:sample_size_per_condition])
    return sampled


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200,
        help="Max UNKNOWN probes to validate per condition.")
    parser.add_argument("--conditions", nargs="+",
        default=["original", "shuffle", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-decile", type=float, default=0.0,
        help="Skip deciles below this (e.g. 0.1 skips the pre-reasoning point).")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--mass-threshold", type=float,
        default=LOST_DATA_MASS_THRESHOLD,
        help="Minimum letter-mass for a correct-argmax UNKNOWN to count as lost data.")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    unknowns = collect_unknown_probes(args.conditions, args.min_decile)
    print(f"Total UNKNOWN probes (decile >= {args.min_decile}): {len(unknowns)}")
    sampled = stratified_sample(unknowns, args.sample_size, rng)
    print(f"Sampling {len(sampled)} probes across {len(args.conditions)} conditions.")

    print(f"Loading {args.model_id} (bfloat16, cuda)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    letter_ids = find_letter_token_ids(tokenizer)
    letter_id_tensor = torch.tensor(
        [letter_ids[L] for L in LABELS], device="cuda"
    )

    results = []
    with torch.inference_mode():
        for i, u in enumerate(sampled):
            prefix_text = (u["reasoning_prefix"] or "").strip()
            asst_prefill = f"<think>{prefix_text}</think>\n"
            text = build_ollama_gemma_prompt(
                BASELINE_SYSTEM_PROMPT, u["prompt"], asst_prefill,
            )
            inputs = tokenizer(
                text, return_tensors="pt", add_special_tokens=True,
            ).to("cuda")
            logits = model(**inputs).logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            letter_probs_tensor = probs[letter_id_tensor]
            letter_total = float(letter_probs_tensor.sum().item())
            letter_probs = {
                L: float(letter_probs_tensor[idx].item())
                for idx, L in enumerate(LABELS)
            }
            argmax_letter = LABELS[int(letter_probs_tensor.argmax().item())]
            argmax_prob = letter_probs[argmax_letter]
            actual = u["actual_answer_label"]
            argmax_correct = argmax_letter == actual
            top1_token_id = int(probs.argmax().item())
            top1_token_prob = float(probs.max().item())
            top1_token_text = tokenizer.decode([top1_token_id])
            lost_data = argmax_correct and letter_total >= args.mass_threshold
            results.append({
                **u,
                "letter_probs": letter_probs,
                "letter_total_mass": letter_total,
                "argmax_letter": argmax_letter,
                "argmax_prob": argmax_prob,
                "argmax_correct": argmax_correct,
                "next_token_top1_id": top1_token_id,
                "next_token_top1_prob": top1_token_prob,
                "next_token_top1_text": top1_token_text,
                "unknown_justified": not lost_data,
                "mass_threshold": args.mass_threshold,
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(sampled):
                print(f"  {i+1}/{len(sampled)}")

    write_jsonl(args.output_path, results)
    print(f"\nWrote {len(results)} rows to {args.output_path}")

    n = len(results)
    if n == 0:
        return
    lost = sum(1 for r in results if not r["unknown_justified"])
    justified = n - lost
    mean_mass = sum(r["letter_total_mass"] for r in results) / n
    argmax_correct_n = sum(1 for r in results if r["argmax_correct"])

    print("\n=== Summary ===")
    print(f"  probes validated:            {n}")
    print(f"  mean letter total mass:      {mean_mass:.3f}")
    print(f"  argmax letter == correct:    {argmax_correct_n}/{n} ({100*argmax_correct_n/n:.1f}%)")
    print(f"  UNKNOWN justified:           {justified}/{n} ({100*justified/n:.1f}%)")
    print(f"  lost data (argmax correct &  mass>=thr={args.mass_threshold}): {lost}/{n} ({100*lost/n:.1f}%)")

    for cond in sorted(set(r["condition"] for r in results)):
        crs = [r for r in results if r["condition"] == cond]
        cn = len(crs)
        cj = sum(1 for r in crs if r["unknown_justified"])
        cmass = sum(r["letter_total_mass"] for r in crs) / cn
        print(f"  {cond:10s}: {cj}/{cn} justified ({100*cj/cn:.1f}%), mean_mass={cmass:.3f}")


if __name__ == "__main__":
    main()
