"""Figure for the decision-level diagnosis on Stacking: share of agent runs on hard levels
that are still solvable and in which no failure has been exposed yet, by action index
(base prompts), and success vs. action budget for agents and search baselines.

    python chain_analysis/plot_fatal_curves.py --out ../ICLR26_CHAIN/Fig/fatal_curves.pdf
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chain_analysis.stacking_oracle.exp2_stacking import success_curve

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = [("Opus base", "Claude-Opus-5.5"), ("Sonnet base", "Claude-Sonnet-5.5"),
          ("sol base", "GPT-6.1-sol"), ("terra base", "GPT-6.1-terra"), ("luna base", "GPT-6-luna")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical order
STYLES = ["-", "--", "-", "--", ":"]
MARKERS = ["o", "s", "^", "D", "v"]


def runs(family, pkg):
    """[(model, fatal_step or None, exposed_step or None)] on hard levels."""
    out = []
    if pkg is None:
        for l in open(os.path.join(HERE, "stacking_oracle/results/exp1_cc/episodes.jsonl")):
            e = json.loads(l)
            if e["difficulty"] == "hard":
                out.append((e["model"], e["fatal_step"], e["exposure_step"] if e["fatal_step"] else None))
    else:
        for l in open(os.path.join(HERE, f"exp1_vs_agents/{pkg}/steps.jsonl")):
            e = json.loads(l)
            if e.get("tier", "hard") == "hard":
                out.append((e["model"], e["fatal_step"], e["exposed_step"] if e["fatal_step"] else None))
    return out


def curve(vals, T):
    """share of runs whose event has not happened by action t (t = 0..T)."""
    return [100 * sum(v is None or v > t for v in vals) / len(vals) for t in range(T + 1)]


CC = os.path.abspath(os.path.join(HERE, "..", "..", "cc_runs"))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GDIRS = {"Opus base": "hard45_opus_base", "Sonnet base": "hard45_sonnet_base",
         "sol base": "hard45_codex_base_gpt-6.1-sol", "terra base": "hard45_codex_base_gpt-6.1-terra",
         "luna base": "hard45_codex_base_gpt-6-luna"}
SEARCH = [("random", "random search", ":"), ("best", "best search", "--")]
GRAY = "#55554f"


def stacking_budget(ax):
    names = sorted({json.loads(l)["level"] for l in open(os.path.join(HERE, "stacking_oracle/results/runs.jsonl"))})
    runs = [json.loads(l) for l in open(os.path.join(HERE, "stacking_oracle/results/runs.jsonl"))]
    B = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
    for (m, label, ls), meth in zip(SEARCH, ("random", "dfs_prune")):
        c = success_curve([r for r in runs if r["method"] == meth], names, B)
        ax.plot(B, [100 * c[b] for b in B], color=GRAY, ls=ls, lw=1.4, label=label)
    eps = [json.loads(l) for l in open(os.path.join(HERE, "stacking_oracle/results/traj_cc.jsonl"))]
    for k, (key, name) in enumerate(AGENTS):
        mine = [e for e in eps if e["model"] == key]
        xs = [b for b in B if b <= 30]
        ax.plot(xs, [100 * sum(e["win"] and e["n_actions"] <= b for e in mine) / len(names) for b in xs],
                color=COLORS[k], ls=STYLES[k], marker=MARKERS[k], ms=3.5, markevery=[len(xs) - 1], lw=1.6)
    ax.set_xscale("log"); ax.set_xlim(1, 1000); ax.set_title("Stacking (77)")
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"]); ax.minorticks_off()
    ax.set_xlabel("action budget"); ax.set_ylabel("success (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    T = 30
    rs = runs("Stacking", None)
    for k, (key, name) in enumerate(AGENTS):
        mine = [r for r in rs if r[0] == key]
        for i, idx in enumerate((1, 2)):
            axes[i].plot(range(T + 1), curve([r[idx] for r in mine], T), color=COLORS[k], ls=STYLES[k],
                         lw=1.5, marker=MARKERS[k], markevery=6, ms=3.2, label=name)
    n = sum(r[0] == AGENTS[0][0] for r in rs)
    axes[0].set_title(f"Still solvable ({n} hard)"); axes[0].set_ylabel("runs (%)")
    axes[1].set_title(f"No failure exposed ({n} hard)"); axes[1].tick_params(labelleft=False)
    for ax in axes[:2]:
        ax.set_xlim(0, T); ax.set_xlabel("action")
    stacking_budget(axes[2])
    axes[2].axvline(30, color="#b4b4ae", lw=0.8, zorder=0)
    axes[2].text(33, 97, "agent budget", fontsize=6.5, color="#55554f", va="top")
    axes[2].set_title("Success vs. budget (all 77)"); axes[2].set_ylabel("")
    for ax in axes:
        ax.set_ylim(0, 102); ax.grid(axis="y", color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(colors="#55554f", length=2)
    axes[2].set_ylabel("success (%)")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    H = [h[0], h2[0], h[1], h2[1]] + h[2:]; L = [l[0], l2[0], l[1], l2[1]] + l[2:]
    fig.legend(H, L, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.36, wspace=0.42)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight")
    for key, name in AGENTS:
        mine = [r for r in rs if r[0] == key]
        print(name, len(mine), "solvable@30", round(curve([r[1] for r in mine], T)[T]),
              "unexposed@30", round(curve([r[2] for r in mine], T)[T]))


if __name__ == "__main__":
    main()
