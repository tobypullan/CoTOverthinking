"""Inspect top right->wrong features: show the questions that activate them,
co-activation patterns, and per-question signatures."""

import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from options_experiment_utils import read_jsonl  # noqa: E402

PER_Q = SCRIPT_DIR / "resample_results" / "sae_l17_w16k_l0medium_per_question.jsonl"
STATS = SCRIPT_DIR / "resample_results" / "sae_l17_w16k_l0medium_feature_stats.jsonl"
RESAMPLE = SCRIPT_DIR / "resample_results" / "options_results_ollama.jsonl"
OUT = SCRIPT_DIR / "resample_results" / "sae_l17_w16k_l0medium_interpretation.md"

TOP_K_FEATURES = 12
TOP_QUESTIONS_PER_FEATURE = 5
NEURONPEDIA = "https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k"


def main():
    per_q = list(read_jsonl(PER_Q))
    stats = list(read_jsonl(STATS))
    q_meta = {row["question_id"]: row for row in read_jsonl(RESAMPLE)}

    # Top-K features over-active in right->wrong
    rw_features = sorted(
        [s for s in stats if s["diff"] > 0],
        key=lambda s: s["cohens_d"], reverse=True,
    )[:TOP_K_FEATURES]

    # Build: feature -> list of (activation, question_id, class)
    feat_activations = defaultdict(list)
    qid_to_class = {r["question_id"]: r["class"] for r in per_q}
    qid_to_active = {r["question_id"]: dict(r["active_features"]) for r in per_q}

    for r in per_q:
        active = dict(r["active_features"])
        for f in rw_features:
            v = active.get(f["feature"], 0.0)
            if v > 0:
                feat_activations[f["feature"]].append((v, r["question_id"], r["class"]))

    lines = []
    lines.append("# Interpreting the top right->wrong features")
    lines.append("")
    lines.append(f"Model: `google/gemma-3-4b-it`, SAE `gemma-scope-2-4b-it-res/layer_17_width_16k_l0_medium`.")
    lines.append(f"Contrast: 108 right->wrong vs 396 right->right MMLU-Pro math questions.")
    lines.append("")
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append("- **Cohen's d** is the between-group effect size in units of pooled standard deviation. "
                 "Rough interpretation: 0.2 = small, 0.5 = medium, 0.8 = large. "
                 "Our top feature is d = +0.80, and ~10 features sit in the medium-large band (0.5-0.7).")
    lines.append("- **active_RW / active_RR** is the fraction of questions in each group where the feature "
                 "fires at all. This separates two very different feature profiles:")
    lines.append("    - *Always-on amplitude shifters* (e.g. feat 279, 589): fire in ~100% of both groups, "
                 "just harder in right->wrong. Probably generic `math-problem-ness` directions that are "
                 "slightly stronger on the harder questions.")
    lines.append("    - *Sparse trigger features* (e.g. feat 4103 — 11% vs 1%, feat 2072 — 17% vs 3%, "
                 "feat 1659 — 7% vs 1%): fire only on a subset of questions, and that subset is heavily "
                 "skewed to right->wrong. These are the interesting candidates for a semantic "
                 "interpretation, because they mark specific kinds of question.")
    lines.append("- **Multiple-testing caveat**: we tested 16384 features, so some separation is expected "
                 "by chance. Cohen's d > 0.5 with N=504 is unlikely to be noise, but for individual "
                 "features you should treat the d value as a *ranking* rather than a calibrated p-value.")
    lines.append("- **Class imbalance caveat**: n(RW)=108 vs n(RR)=396. Means for RR are therefore more "
                 "stable; a rarely-firing feature with only ~1% active rate in RR has very few samples to "
                 "estimate its mean from — big relative differences can be partly noise.")
    lines.append("- **Not evidence of causation**: these features *correlate with* questions the model "
                 "gets right early then abandons. They don't prove that suppressing the feature would "
                 "fix the flip. Test that with an activation patching / steering experiment.")
    lines.append("")
    lines.append("## Patterns in the top 12")
    lines.append("")
    lines.append("The top-12 right->wrong features split into roughly three types (eyeballing the table):")
    lines.append("")
    lines.append("1. **Universal-but-stronger** (active_RW ≈ active_RR ≈ 1.00): feats 279, 589, 337, "
                 "166, 6055. These are always on; the model simply activates them harder on harder "
                 "questions. Low interpretive value on their own — they probably encode 'this is a math "
                 "word problem' or similar.")
    lines.append("2. **Moderately selective** (active_RW ~ 0.4-0.6, active_RR ~ 0.2-0.4): feats 471, "
                 "346, 565, 598, 10753. These fire on about half of right->wrong questions and half as "
                 "often on right->right. They're the best candidates for topic-level features (e.g. a "
                 "specific type of algebra / probability wording).")
    lines.append("3. **Sharp selective triggers** (active_RW < 0.2, but 5-10× the RR rate): feats 4103, "
                 "2072, 1659, 4551, 3871. These fire rarely, but when they do the question is much more "
                 "likely to be a right->wrong case. These are the best targets for Neuronpedia inspection.")
    lines.append("")
    lines.append("## Top questions per feature")
    lines.append("")
    lines.append("For each feature, the right->wrong questions where it fires hardest. Compare the question "
                 "prompts — shared vocabulary / structure across the top questions is your interpretation "
                 "signal.")
    lines.append("")

    for f in rw_features:
        fid = f["feature"]
        acts = feat_activations[fid]
        # split RW and RR, show top RW
        rw_acts = sorted([a for a in acts if a[2] == "right_wrong"], reverse=True)
        rr_acts = sorted([a for a in acts if a[2] == "right_right"], reverse=True)
        lines.append(f"### Feature {fid} — d={f['cohens_d']:+.3f}, "
                     f"active {f['active_rw']:.2f} vs {f['active_rr']:.2f}, "
                     f"[Neuronpedia]({NEURONPEDIA}/{fid})")
        lines.append("")
        lines.append(f"Top right->wrong activations:")
        lines.append("")
        for act_val, qid, _cls in rw_acts[:TOP_QUESTIONS_PER_FEATURE]:
            meta = q_meta.get(qid, {})
            q_text = (meta.get("question") or "").strip().replace("\n", " ")
            q_text = q_text[:220] + ("..." if len(q_text) > 220 else "")
            cat = meta.get("category", "?")
            lines.append(f"- **{act_val:.1f}** — qid={qid} [{cat}]: {q_text}")
        if rr_acts:
            lines.append("")
            lines.append(f"For contrast, top right->right activations of the same feature:")
            lines.append("")
            for act_val, qid, _cls in rr_acts[:3]:
                meta = q_meta.get(qid, {})
                q_text = (meta.get("question") or "").strip().replace("\n", " ")
                q_text = q_text[:180] + ("..." if len(q_text) > 180 else "")
                lines.append(f"- {act_val:.1f} — qid={qid}: {q_text}")
        lines.append("")

    # Co-activation: for each pair of top features, how many RW questions have both
    lines.append("## Co-activation among top features (right->wrong questions only)")
    lines.append("")
    lines.append("How often do pairs of top features fire on the same right->wrong question. "
                 "High co-activation means the features are probably picking up related concepts "
                 "(or the same direction viewed through different codes).")
    lines.append("")
    top_ids = [f["feature"] for f in rw_features]
    rw_qids = [r["question_id"] for r in per_q if r["class"] == "right_wrong"]
    active_sets = {fid: set() for fid in top_ids}
    for qid in rw_qids:
        active = qid_to_active.get(qid, {})
        for fid in top_ids:
            if fid in active:
                active_sets[fid].add(qid)
    lines.append("| feature |" + " | ".join(str(f) for f in top_ids) + " |")
    lines.append("|--" + "|--".join(["-"] * (len(top_ids) + 1)) + "|")
    for fid_a in top_ids:
        row = [str(fid_a)]
        for fid_b in top_ids:
            if fid_a == fid_b:
                row.append(str(len(active_sets[fid_a])))
            else:
                inter = len(active_sets[fid_a] & active_sets[fid_b])
                row.append(str(inter))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"(Diagonal = number of right->wrong questions on which that feature fires at all; "
                 f"total right->wrong questions = {len(rw_qids)}.)")
    lines.append("")

    # How many of the top features fire per RW question — a simple "RW signature"
    signature_counts = defaultdict(int)
    for qid in rw_qids:
        active = qid_to_active.get(qid, {})
        n_hits = sum(1 for fid in top_ids if fid in active)
        signature_counts[n_hits] += 1
    rr_qids = [r["question_id"] for r in per_q if r["class"] == "right_right"]
    signature_counts_rr = defaultdict(int)
    for qid in rr_qids:
        active = qid_to_active.get(qid, {})
        n_hits = sum(1 for fid in top_ids if fid in active)
        signature_counts_rr[n_hits] += 1
    lines.append("## Signature: how many of the top-12 RW features fire per question")
    lines.append("")
    lines.append("| # top-12 features active | right->wrong | right->right |")
    lines.append("|---|---|---|")
    max_k = max(list(signature_counts.keys()) + list(signature_counts_rr.keys()))
    for k in range(max_k + 1):
        nrw = signature_counts.get(k, 0)
        nrr = signature_counts_rr.get(k, 0)
        pct_rw = 100 * nrw / len(rw_qids)
        pct_rr = 100 * nrr / len(rr_qids)
        lines.append(f"| {k} | {nrw} ({pct_rw:.0f}%) | {nrr} ({pct_rr:.0f}%) |")
    lines.append("")
    lines.append("A right->wrong question is likely to light up several of these features at once; "
                 "a right->right question rarely lights up more than a couple. Whether this is "
                 "predictive out-of-sample needs a held-out split.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
