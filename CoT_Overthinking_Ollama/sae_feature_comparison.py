"""Compare gemma-scope-2 SAE features between right->wrong and right->right questions.

For each question in the `original` resample condition we classify its
per-question trajectory (using decile <1.0 correctness vs decile 1.0 correctness),
then encode the question's prompt through gemma-3-4b-it, capture the residual
stream at a chosen layer/position, push it through a gemma-scope-2 residual
SAE, and collect the resulting sparse feature activations.

We then compute per-feature statistics (mean activation, active rate, Welch t,
Cohen's d) between the right->wrong and right->right groups, rank features by
|Cohen's d|, and write a top-features report including Neuronpedia links.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import read_jsonl, write_jsonl  # noqa: E402
from ollama_options import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT  # noqa: E402

MODEL_ID = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
DEFAULT_LAYER = 17
DEFAULT_WIDTH = "16k"
DEFAULT_L0 = "medium"

RESAMPLE_PATH = SCRIPT_DIR / "resample_results" / "options_results_ollama.jsonl"
OUT_DIR = SCRIPT_DIR / "resample_results"


def build_prompt(system_text, user_text):
    """Match the Ollama gemma3 chat template used everywhere else."""
    return (
        f"<start_of_turn>user\n{system_text}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def classify_trajectories(resample_path):
    """Return {question_id: (class, prompt)} for right->wrong and right->right."""
    out = {}
    for row in read_jsonl(resample_path):
        qid = row["question_id"]
        prompt = row["prompt"]
        results = {float(p["resample_point"]): p for p in row["resample_results"]}
        final = results.get(1.0)
        if final is None:
            continue
        final_correct = bool(final["correct"])
        ever_correct_before = any(
            bool(p["correct"])
            for d, p in results.items()
            if d < 1.0 and p["parsed_answer"] != "UNKNOWN"
        )
        if final_correct and ever_correct_before:
            cls = "right_right"
        elif (not final_correct) and ever_correct_before:
            cls = "right_wrong"
        else:
            continue  # skip wrong->right and wrong->wrong for this contrast
        out[qid] = (cls, prompt)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        choices=[9, 17, 22, 29])
    parser.add_argument("--width", type=str, default=DEFAULT_WIDTH,
                        choices=["16k", "65k", "262k", "1m"])
    parser.add_argument("--l0", type=str, default=DEFAULT_L0,
                        choices=["small", "medium", "big"])
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="File prefix for outputs; defaults to layer/width/l0.")
    return parser.parse_args()


def welch_t(mean_a, var_a, n_a, mean_b, var_b, n_b):
    denom = math.sqrt(var_a / n_a + var_b / n_b) if n_a and n_b else 0.0
    if denom == 0.0:
        return 0.0
    return (mean_a - mean_b) / denom


def cohens_d(mean_a, var_a, n_a, mean_b, var_b, n_b):
    if n_a + n_b <= 2:
        return 0.0
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return (mean_a - mean_b) / math.sqrt(pooled_var)


def main():
    args = parse_args()
    sae_id = f"layer_{args.layer}_width_{args.width}_l0_{args.l0}"
    prefix = args.output_prefix or f"sae_l{args.layer}_w{args.width}_l0{args.l0}"
    features_path = OUT_DIR / f"{prefix}_per_question.jsonl"
    report_path = OUT_DIR / f"{prefix}_top_features.md"

    print(f"Classifying trajectories from {RESAMPLE_PATH.name}")
    trajectories = classify_trajectories(RESAMPLE_PATH)
    class_counts = {}
    for cls, _ in trajectories.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"  class counts: {class_counts}")

    print(f"Loading {MODEL_ID} (bfloat16, cuda)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    print(f"Loading SAE release={SAE_RELEASE} sae_id={sae_id}...")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device="cuda")
    neuronpedia_id = sae.cfg.metadata.get("neuronpedia_id", None)
    print(f"  SAE d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} neuronpedia={neuronpedia_id}")
    # gemma-3 uses Gemma3Config (multimodal wrapper); text cfg is .text_config
    text_cfg = getattr(model.config, "text_config", model.config)
    assert sae.cfg.d_in == text_cfg.hidden_size, "SAE d_in != model hidden size"

    hidden_index = args.layer + 1  # hidden_states[0]=embeddings, [L+1]=after block L

    d_sae = sae.cfg.d_sae
    # per-class running stats: sum, sum_of_squares, active_count, total_count
    stats = {c: {
        "sum":    torch.zeros(d_sae, dtype=torch.float64, device="cuda"),
        "sq_sum": torch.zeros(d_sae, dtype=torch.float64, device="cuda"),
        "active": torch.zeros(d_sae, dtype=torch.float64, device="cuda"),
        "n": 0,
    } for c in ("right_wrong", "right_right")}

    per_question_rows = []
    items = sorted(trajectories.items(), key=lambda kv: kv[0])
    n_total = len(items)

    with torch.inference_mode():
        for i, (qid, (cls, prompt)) in enumerate(items):
            text = build_prompt(BASELINE_SYSTEM_PROMPT, prompt)
            inputs = tokenizer(text, return_tensors="pt",
                               add_special_tokens=True).to("cuda")
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            resid = outputs.hidden_states[hidden_index][0, -1, :].float()
            feats = sae.encode(resid.to(sae.dtype).unsqueeze(0))[0].float()

            s = stats[cls]
            s["sum"] += feats.double()
            s["sq_sum"] += (feats.double() ** 2)
            s["active"] += (feats > 0).double()
            s["n"] += 1

            nz = torch.nonzero(feats, as_tuple=False).flatten().tolist()
            per_question_rows.append({
                "question_id": qid,
                "class": cls,
                "n_active": len(nz),
                "active_features": [
                    [int(idx), float(feats[idx].item())] for idx in nz
                ],
            })

            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                print(f"  {i+1}/{n_total}")

    write_jsonl(features_path, per_question_rows)
    print(f"Wrote per-question features to {features_path}")

    # Compute per-feature stats
    rw = stats["right_wrong"]
    rr = stats["right_right"]
    n_rw = rw["n"]
    n_rr = rr["n"]
    mean_rw = (rw["sum"] / max(n_rw, 1)).cpu()
    mean_rr = (rr["sum"] / max(n_rr, 1)).cpu()
    var_rw = (rw["sq_sum"] / max(n_rw, 1) - mean_rw.to(rw["sq_sum"].device) ** 2).clamp(min=0).cpu()
    var_rr = (rr["sq_sum"] / max(n_rr, 1) - mean_rr.to(rr["sq_sum"].device) ** 2).clamp(min=0).cpu()
    active_rw = (rw["active"] / max(n_rw, 1)).cpu()
    active_rr = (rr["active"] / max(n_rr, 1)).cpu()

    mean_rw = mean_rw.numpy()
    mean_rr = mean_rr.numpy()
    var_rw = var_rw.numpy()
    var_rr = var_rr.numpy()
    active_rw = active_rw.numpy()
    active_rr = active_rr.numpy()

    rows = []
    for f in range(d_sae):
        if mean_rw[f] == 0 and mean_rr[f] == 0:
            continue
        d = cohens_d(mean_rw[f], var_rw[f], n_rw,
                     mean_rr[f], var_rr[f], n_rr)
        t = welch_t(mean_rw[f], var_rw[f], n_rw,
                    mean_rr[f], var_rr[f], n_rr)
        rows.append({
            "feature": f,
            "mean_rw": float(mean_rw[f]),
            "mean_rr": float(mean_rr[f]),
            "active_rw": float(active_rw[f]),
            "active_rr": float(active_rr[f]),
            "cohens_d": float(d),
            "welch_t": float(t),
            "diff": float(mean_rw[f] - mean_rr[f]),
        })
    rows.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)

    # Write full feature table
    write_jsonl(OUT_DIR / f"{prefix}_feature_stats.jsonl", rows)

    # Markdown report
    top_rw = [r for r in rows if r["diff"] > 0][: args.top_k]
    top_rr = [r for r in rows if r["diff"] < 0][: args.top_k]
    lines = []
    lines.append(f"# SAE feature contrast: right->wrong vs right->right")
    lines.append("")
    lines.append(f"- Model: `{MODEL_ID}`")
    lines.append(f"- SAE: `{SAE_RELEASE}` / `{sae_id}` ({d_sae} features)")
    lines.append(f"- Hook: `{sae.cfg.metadata.get('hook_name')}` (layer {args.layer} residual)")
    lines.append(f"- Position: last prompt token (end of `<start_of_turn>model\\n`)")
    lines.append(f"- n(right->wrong) = {n_rw}, n(right->right) = {n_rr}")
    lines.append(f"- Neuronpedia: https://neuronpedia.org/{neuronpedia_id}/<feature>")
    lines.append("")
    lines.append(f"## Top {args.top_k} features higher in right->wrong (over-active when model later flips)")
    lines.append("")
    lines.append("| feature | cohen_d | mean_RW | mean_RR | active_RW | active_RR | link |")
    lines.append("|---------|---------|---------|---------|-----------|-----------|------|")
    for r in top_rw:
        link = f"https://neuronpedia.org/{neuronpedia_id}/{r['feature']}" if neuronpedia_id else ""
        lines.append(
            f"| {r['feature']} | {r['cohens_d']:+.3f} | {r['mean_rw']:.3f} | "
            f"{r['mean_rr']:.3f} | {r['active_rw']:.2f} | {r['active_rr']:.2f} | {link} |"
        )
    lines.append("")
    lines.append(f"## Top {args.top_k} features higher in right->right (absent when model flips)")
    lines.append("")
    lines.append("| feature | cohen_d | mean_RW | mean_RR | active_RW | active_RR | link |")
    lines.append("|---------|---------|---------|---------|-----------|-----------|------|")
    for r in top_rr:
        link = f"https://neuronpedia.org/{neuronpedia_id}/{r['feature']}" if neuronpedia_id else ""
        lines.append(
            f"| {r['feature']} | {r['cohens_d']:+.3f} | {r['mean_rw']:.3f} | "
            f"{r['mean_rr']:.3f} | {r['active_rw']:.2f} | {r['active_rr']:.2f} | {link} |"
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote report to {report_path}")

    # Brief stdout summary
    print("\n=== Top features over-active in right->wrong ===")
    for r in top_rw[:10]:
        print(f"  feat {r['feature']:>6d}  d={r['cohens_d']:+.3f}  "
              f"mean RW={r['mean_rw']:.3f} RR={r['mean_rr']:.3f}  "
              f"active RW={r['active_rw']:.2f} RR={r['active_rr']:.2f}")
    print("\n=== Top features over-active in right->right ===")
    for r in top_rr[:10]:
        print(f"  feat {r['feature']:>6d}  d={r['cohens_d']:+.3f}  "
              f"mean RW={r['mean_rw']:.3f} RR={r['mean_rr']:.3f}  "
              f"active RW={r['active_rw']:.2f} RR={r['active_rr']:.2f}")


if __name__ == "__main__":
    main()
