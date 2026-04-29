"""Figure: questions the model got correct at >1 decile, then wrong at decile 1.0."""

import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "resample_results" / "options_results_ollama.jsonl"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

CORRECT, WRONG, UNK = 2, 1, 0
COLOURS = ["#bdbdbd", "#d62728", "#2ca02c"]
LABELS = {UNK: "UNKNOWN", WRONG: "wrong letter", CORRECT: "correct"}


OPTION_RE = re.compile(r"\n([A-J]): ")


def load():
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            r = json.loads(line)
            rs = sorted(r["resample_results"], key=lambda x: float(x["resample_point"]))
            options = sorted(set(OPTION_RE.findall(r["prompt"])))
            rows.append({"qid": r["question_id"], "rs": rs, "options": options})
    return rows


def classify(pr):
    if pr["correct"]:
        return CORRECT
    if pr["parsed_answer"] == "UNKNOWN":
        return UNK
    return WRONG


def select_matching(rows):
    """Questions with >1 correct deciles AND wrong at the final decile."""
    out = []
    for r in rows:
        codes = [classify(pr) for pr in r["rs"]]
        n_correct = sum(c == CORRECT for c in codes)
        final_correct = codes[-1] == CORRECT
        if n_correct > 1 and not final_correct:
            out.append({"qid": r["qid"], "codes": codes, "n_correct": n_correct})
    return out


def make_figure(matching, total_q, test_result=None, cond_test_result=None):
    deciles = np.arange(0.0, 1.01, 0.1)
    # sort: most correct first; tiebreak by latest correct decile (so the "drop"
    # at the right-hand side is visually grouped)
    matching = sorted(
        matching,
        key=lambda m: (-m["n_correct"],
                       -max(i for i, c in enumerate(m["codes"]) if c == CORRECT)),
    )
    grid = np.array([m["codes"] for m in matching])

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[5, 2.2],
                          hspace=0.45, wspace=0.32)

    # --- Heatmap: trajectories ---
    ax_h = fig.add_subplot(gs[0, 0])
    cmap = ListedColormap(COLOURS)
    ax_h.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5,
                interpolation="nearest")
    ax_h.set_xticks(range(len(deciles)))
    ax_h.set_xticklabels([f"{d:.1f}" for d in deciles])
    ax_h.set_xlabel("Decile of CoT prefix injected")
    ax_h.set_ylabel(f"Question (n={len(matching)}, sorted by # correct)")
    ax_h.set_title(
        f"Per-decile outcome — {len(matching)} questions correct >1× then "
        f"WRONG at decile 1.0"
    )
    # mark the final column
    ax_h.axvline(len(deciles) - 1.5, color="black", lw=1.2, alpha=0.7)
    ax_h.annotate(
        "final answer",
        xy=(len(deciles) - 1, len(matching) - 0.5),
        xytext=(len(deciles) - 1, len(matching) + 1.2),
        ha="center", va="top", fontsize=9, color="black",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )
    legend_h = [Patch(facecolor=COLOURS[k], edgecolor="black", label=LABELS[k])
                for k in (CORRECT, WRONG, UNK)]
    ax_h.legend(handles=legend_h, loc="upper left",
                bbox_to_anchor=(0.0, -0.08), ncol=3, frameon=False)

    # --- Bar chart: distribution of #correct deciles among matching ---
    ax_b = fig.add_subplot(gs[0, 1])
    counter = Counter(m["n_correct"] for m in matching)
    xs = sorted(counter)
    ys = [counter[x] for x in xs]
    bars = ax_b.barh(xs, ys, color="#2ca02c", alpha=0.85)
    ax_b.set_yticks(xs)
    ax_b.set_xlabel("Questions")
    ax_b.set_ylabel("# correct deciles before final wrong")
    ax_b.set_title("Distribution")
    ax_b.invert_yaxis()
    for bar, v in zip(bars, ys):
        ax_b.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                  str(v), va="center", fontsize=10)
    ax_b.set_xlim(0, max(ys) * 1.25)
    ax_b.grid(axis="x", alpha=0.3)

    # --- Footer panel: headline stats as text ---
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis("off")
    n_match = len(matching)
    pct = 100 * n_match / total_q
    header = (
        f"$\\bf{{{n_match}/{total_q}}}$ ({pct:.1f}%) questions had the correct "
        f"answer at >1 mid-CoT decile but were WRONG at the full-reasoning "
        f"(decile 1.0) answer.\n"
        f"Model: gemma3:4b   ·   Dataset: MMLU-Pro math   ·   "
        f"11 deciles per question (0.0 → 1.0)"
    )
    ax_t.text(0.5, 0.95, header, ha="center", va="top", fontsize=12,
              transform=ax_t.transAxes)

    # Two-column comparison of the two hypothesis tests
    if test_result is not None and cond_test_result is not None:
        tr, cr = test_result, cond_test_result

        def block(ax, x, title, tr, denom_text, verdict, edge, fill):
            body = (
                f"denominator: {denom_text}\n"
                f"observed = {tr['observed']}\n"
                f"expected (H₀) = {tr['expected']:.1f} ± {tr['sd']:.1f}\n"
                f"z = {tr['z']:+.2f},   MC p(X ≥ obs) = {tr['p_value_mc']:.3f}"
            )
            ax.text(x, 0.62, title, ha="left", va="top", fontsize=11.5,
                    fontweight="bold", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=fill,
                              edgecolor=edge, linewidth=1.2))
            ax.text(x, 0.42, body, ha="left", va="top", fontsize=10.5,
                    transform=ax.transAxes, color="#222222")
            ax.text(x, 0.05, verdict, ha="left", va="bottom", fontsize=10.5,
                    fontweight="bold", color=edge, transform=ax.transAxes)

        block(
            ax_t, 0.02,
            "Unconditional test",
            tr,
            f"all {total_q} questions",
            "not significant — pattern is below chance",
            edge="#555555", fill="#f0f0f0",
        )
        block(
            ax_t, 0.52,
            "Conditional on wrong-at-final",
            cr,
            f"{cr['n_wrong_final']} failures "
            f"({cr['n_unknown_final']} UNKNOWN-final)",
            f"p ≈ {cr['p_value_mc']:.2f}  —  modestly elevated above chance",
            edge="#b8001f", fill="#ffecec",
        )

    out_path = OUT_DIR / "right_to_wrong.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, n_match


def hypothesis_test(rows, observed, n_sim=20000, seed=0):
    """Test H0: the apparent 'correct >1× then wrong at 1.0' pattern is what
    you'd see if the model's parsed answers were unrelated to the truth.

    Per-question null: condition on the model's emitted sequence; draw the
    correct answer uniformly from that question's option set. The pattern
    triggers iff some option o satisfies (count(seq == o) >= 2 AND seq[-1] != o)
    AND the random truth equals that o, giving per-question probability
        p_q = |{o in options : count(o) >= 2 and o != final}| / N_options.
    Total matches ~ Poisson-Binomial(p_1, ..., p_Q). We report normal-approx
    and Monte Carlo p-values for P(X >= observed).
    """
    p_per_q = []
    for r in rows:
        opts = r["options"]
        if not opts:
            continue
        seq = [pr["parsed_answer"] for pr in r["rs"]]
        final = seq[-1]
        counts = Counter(seq)
        n_satisfying = sum(1 for o in opts if counts.get(o, 0) >= 2 and o != final)
        p_per_q.append(n_satisfying / len(opts))

    p = np.array(p_per_q)
    expected = float(p.sum())
    var = float((p * (1 - p)).sum())
    sd = math.sqrt(var)

    # normal approximation with continuity correction
    z = (observed - 0.5 - expected) / sd
    p_normal = 0.5 * math.erfc(z / math.sqrt(2))

    # Monte Carlo Poisson-Binomial
    rng = np.random.default_rng(seed)
    sim = (rng.random((n_sim, len(p))) < p).sum(axis=1)
    p_mc = float((sim >= observed).mean())
    sim_mean, sim_sd = float(sim.mean()), float(sim.std())

    return {
        "n_questions": len(p),
        "observed": observed,
        "expected": expected,
        "sd": sd,
        "z": z,
        "p_value_normal": p_normal,
        "p_value_mc": p_mc,
        "mc_mean": sim_mean,
        "mc_sd": sim_sd,
        "mc_max": int(sim.max()),
        "n_sim": n_sim,
    }


def hypothesis_test_conditional(rows, n_sim=20000, seed=0):
    """Conditional version of the test: restrict to questions where the final
    answer was wrong, and ask whether the right-then-wrong pattern is more
    common than chance among those failures.

    For each wrong-at-final question, under H0 the truth is uniform over:
      - the N-1 options other than the final letter (if final is a letter), or
      - all N options (if final is UNKNOWN, since UNKNOWN matches no truth).
    Trigger probability per question:
      - final letter L: |{o : count(o) >= 2, o != L}| / (N - 1)
      - final UNKNOWN : |{o : count(o) >= 2}|           /  N
    """
    p_per_q = []
    observed = 0
    n_unknown_final = 0
    for r in rows:
        opts = r["options"]
        if not opts:
            continue
        seq = [pr["parsed_answer"] for pr in r["rs"]]
        final = seq[-1]
        actual = r["rs"][-1]["actual_answer"]
        final_wrong = (final != actual)
        if not final_wrong:
            continue
        counts = Counter(seq)
        n_corr = counts.get(actual, 0)
        if n_corr > 1:
            observed += 1
        if final == "UNKNOWN":
            n_unknown_final += 1
            n_satisfying = sum(1 for o in opts if counts.get(o, 0) >= 2)
            p_per_q.append(n_satisfying / len(opts))
        else:
            n_satisfying = sum(1 for o in opts if counts.get(o, 0) >= 2 and o != final)
            denom = len(opts) - 1
            p_per_q.append(n_satisfying / denom if denom > 0 else 0.0)

    p = np.array(p_per_q)
    expected = float(p.sum())
    var = float((p * (1 - p)).sum())
    sd = math.sqrt(var)
    z = (observed - 0.5 - expected) / sd if sd > 0 else float("nan")
    p_normal = 0.5 * math.erfc(z / math.sqrt(2)) if sd > 0 else float("nan")

    rng = np.random.default_rng(seed)
    sim = (rng.random((n_sim, len(p))) < p).sum(axis=1)
    p_mc = float((sim >= observed).mean())

    return {
        "n_wrong_final": len(p),
        "n_unknown_final": n_unknown_final,
        "observed": observed,
        "expected": expected,
        "sd": sd,
        "z": z,
        "p_value_normal": p_normal,
        "p_value_mc": p_mc,
        "mc_mean": float(sim.mean()),
        "mc_sd": float(sim.std()),
        "mc_max": int(sim.max()),
        "n_sim": n_sim,
    }


def main():
    rows = load()
    matching = select_matching(rows)

    print("--- Hypothesis test (unconditional, all 1000 questions) ---")
    print("H0: model's answers are independent of the truth.")
    res = hypothesis_test(rows, observed=len(matching))

    res2 = None  # will be filled below; figure is rendered after both tests run
    print(f"  questions analysed       : {res['n_questions']}")
    print(f"  observed matches         : {res['observed']}")
    print(f"  expected under H0        : {res['expected']:.2f}")
    print(f"  sd under H0              : {res['sd']:.2f}")
    print(f"  z (continuity-corrected) : {res['z']:.2f}")
    print(f"  p-value (normal approx)  : {res['p_value_normal']:.3e}")
    print(f"  p-value (Monte Carlo)    : {res['p_value_mc']:.3e}  "
          f"[{res['n_sim']} sims, MC mean={res['mc_mean']:.2f} "
          f"sd={res['mc_sd']:.2f} max={res['mc_max']}]")

    print("\n--- Hypothesis test (CONDITIONAL on wrong-at-final) ---")
    print("H0: among questions the model got wrong at the end, the pattern")
    print("    triggers no more often than uniform-guess chance.")
    res2 = hypothesis_test_conditional(rows)
    print(f"  wrong-at-final           : {res2['n_wrong_final']}  "
          f"({res2['n_unknown_final']} of which final=UNKNOWN)")
    print(f"  observed pattern triggers: {res2['observed']}")
    print(f"  expected under H0        : {res2['expected']:.2f}")
    print(f"  sd under H0              : {res2['sd']:.2f}")
    print(f"  z (continuity-corrected) : {res2['z']:.2f}")
    print(f"  p-value (normal approx)  : {res2['p_value_normal']:.3e}")
    print(f"  p-value (Monte Carlo)    : {res2['p_value_mc']:.3e}  "
          f"[{res2['n_sim']} sims, MC mean={res2['mc_mean']:.2f} "
          f"sd={res2['mc_sd']:.2f} max={res2['mc_max']}]")

    out_path, n = make_figure(
        matching, total_q=len(rows), test_result=res, cond_test_result=res2
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
