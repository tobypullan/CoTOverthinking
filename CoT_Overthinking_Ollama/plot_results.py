"""Generate visualisations for results.md sections 3-6."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESAMPLE_DIR = SCRIPT_DIR / "resample_results"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "original": RESAMPLE_DIR / "options_results_ollama.jsonl",
    "shuffle":  RESAMPLE_DIR / "options_shuffle_results_ollama.jsonl",
    "random":   RESAMPLE_DIR / "options_random_results_ollama.jsonl",
}
COLOURS = {"original": "#1f77b4", "shuffle": "#ff7f0e", "random": "#2ca02c"}


def load_decile_stats():
    stats = {}
    for cond, path in CONDITIONS.items():
        bucket = defaultdict(lambda: [0, 0, 0])  # n, unk, correct
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                for pr in row["resample_results"]:
                    d = float(pr["resample_point"])
                    bucket[d][0] += 1
                    if pr["parsed_answer"] == "UNKNOWN":
                        bucket[d][1] += 1
                    if pr["correct"]:
                        bucket[d][2] += 1
        deciles = sorted(bucket.keys())
        stats[cond] = {
            "deciles": deciles,
            "n":       [bucket[d][0] for d in deciles],
            "unk_pct": [100 * bucket[d][1] / bucket[d][0] for d in deciles],
            "acc_pct": [100 * bucket[d][2] / bucket[d][0] for d in deciles],
        }
    return stats


def plot_accuracy_by_decile(stats):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, s in stats.items():
        ax.plot(s["deciles"], s["acc_pct"], marker="o",
                label=cond, color=COLOURS[cond], linewidth=2)
    ax.axhline(10, ls="--", color="grey", alpha=0.6, label="chance (10%)")
    ax.set_xlabel("Decile of CoT prefix injected")
    ax.set_ylabel("Correct %")
    ax.set_title("Probe accuracy by decile (gemma3:4b, MMLU-Pro math, n=1000)")
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_by_decile.png", dpi=150)
    plt.close(fig)


def plot_unknown_by_decile(stats):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, s in stats.items():
        ax.plot(s["deciles"], s["unk_pct"], marker="s",
                label=cond, color=COLOURS[cond], linewidth=2)
    ax.set_xlabel("Decile of CoT prefix injected")
    ax.set_ylabel("UNKNOWN %")
    ax.set_title("UNKNOWN parse rate by decile")
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "unknown_by_decile.png", dpi=150)
    plt.close(fig)


def plot_headline_bar(stats):
    fig, ax = plt.subplots(figsize=(6, 5))
    conds = list(stats.keys())
    d1 = [stats[c]["acc_pct"][-1] for c in conds]
    bars = ax.bar(conds, d1, color=[COLOURS[c] for c in conds])
    ax.axhline(10, ls="--", color="grey", alpha=0.6, label="chance (10%)")
    ax.set_ylabel("Correct % at decile 1.0")
    ax.set_title("Final-answer accuracy — content vs length")
    for bar, v in zip(bars, d1):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "headline_decile1.png", dpi=150)
    plt.close(fig)


def plot_unknown_totals(stats):
    fig, ax = plt.subplots(figsize=(6, 5))
    conds = list(stats.keys())
    totals = [sum(n * p / 100 for n, p in zip(stats[c]["n"], stats[c]["unk_pct"]))
              for c in conds]
    grand = [sum(stats[c]["n"]) for c in conds]
    pct = [100 * t / g for t, g in zip(totals, grand)]
    bars = ax.bar(conds, pct, color=[COLOURS[c] for c in conds])
    ax.set_ylabel("UNKNOWN % of 11,000 probes per condition")
    ax.set_title("Total UNKNOWN rate per condition")
    for bar, v, t in zip(bars, pct, totals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f"{v:.1f}%\n({int(t)})", ha="center", fontsize=10)
    ax.set_ylim(0, 70)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "unknown_totals.png", dpi=150)
    plt.close(fig)


def plot_logit_validation():
    path = RESAMPLE_DIR / "unknown_logit_validation.jsonl"
    by_cond = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_cond[r["condition"]].append(r)

    conds = ["original", "shuffle", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    justified_pct = []
    lost_pct = []
    mean_mass = []
    for c in conds:
        rs = by_cond[c]
        n = len(rs)
        j = sum(1 for r in rs if r["unknown_justified"])
        justified_pct.append(100 * j / n)
        lost_pct.append(100 * (n - j) / n)
        mean_mass.append(sum(r["letter_total_mass"] for r in rs) / n)

    x = np.arange(len(conds))
    axes[0].bar(x, justified_pct, color="#4c72b0", label="justified")
    axes[0].bar(x, lost_pct, bottom=justified_pct, color="#c44e52", label="lost data")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylabel("% of sampled UNKNOWN probes")
    axes[0].set_title("Logit validation of UNKNOWN probes\n(200 per condition)")
    axes[0].set_ylim(0, 105)
    for i, (j, l) in enumerate(zip(justified_pct, lost_pct)):
        axes[0].text(i, j/2, f"{j:.1f}%", ha="center", color="white",
                     fontweight="bold")
        axes[0].text(i, j + l/2, f"{l:.1f}%", ha="center", color="white",
                     fontsize=9)
    axes[0].legend(loc="lower right")

    axes[1].bar(conds, mean_mass, color=[COLOURS[c] for c in conds])
    axes[1].set_ylabel("Mean letter-mass at next token")
    axes[1].set_title("Commit pressure at UNKNOWN positions")
    axes[1].set_ylim(0, 0.6)
    for i, v in enumerate(mean_mass):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "logit_validation.png", dpi=150)
    plt.close(fig)


def plot_right_then_wrong():
    path = CONDITIONS["original"]
    counts = {"right_stayed_right": 0, "right_then_wrong": 0,
              "wrong_then_right": 0, "wrong_stayed_wrong": 0}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            results = {float(pr["resample_point"]): pr for pr in row["resample_results"]}
            final = results.get(1.0) or results.get(1)
            if final is None:
                continue
            final_correct = bool(final["correct"])
            ever_correct_before = any(
                bool(pr["correct"])
                for d, pr in results.items()
                if d < 1.0 and pr["parsed_answer"] != "UNKNOWN"
            )
            if final_correct and ever_correct_before:
                counts["right_stayed_right"] += 1
            elif final_correct and not ever_correct_before:
                counts["wrong_then_right"] += 1
            elif (not final_correct) and ever_correct_before:
                counts["right_then_wrong"] += 1
            else:
                counts["wrong_stayed_wrong"] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["right → right", "right → WRONG", "wrong → right", "wrong → wrong"]
    keys = ["right_stayed_right", "right_then_wrong",
            "wrong_then_right", "wrong_stayed_wrong"]
    values = [counts[k] for k in keys]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Questions (n=1000)")
    ax.set_title("Per-question trajectory (original condition)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 5, str(v),
                ha="center", fontweight="bold")
    final_wrong = counts["right_then_wrong"] + counts["wrong_stayed_wrong"]
    rtw_pct = 100 * counts["right_then_wrong"] / final_wrong
    ax.text(0.98, 0.95,
            f"right→wrong = {counts['right_then_wrong']} / {final_wrong} final-wrong ({rtw_pct:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "right_then_wrong.png", dpi=150)
    plt.close(fig)
    print("right-then-wrong counts:", counts)


def main():
    stats = load_decile_stats()
    plot_accuracy_by_decile(stats)
    plot_unknown_by_decile(stats)
    plot_headline_bar(stats)
    plot_unknown_totals(stats)
    plot_logit_validation()
    plot_right_then_wrong()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
