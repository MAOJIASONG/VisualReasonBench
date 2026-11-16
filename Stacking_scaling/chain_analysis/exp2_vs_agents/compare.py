"""Experiment 2 vs the agent runs: search baselines and Claude Code / Codex on the 45 hard
levels (15 per game), at the same verify budget (and optionally the same 40-action budget).

    python chain_analysis/exp2_vs_agents/compare.py [--action-cap 40] [--max-v 64]
Baseline runs: <game>/analysis_exp2/runs.jsonl; agent runs: ../cc_runs/hard45_*/results_*.jsonl.
"""
import argparse, glob, json, os

GAMES = [("type_3", "FluidNet"), ("type_3_1", "Domino"), ("type_2_3", "PipeFit")]
METHODS = [("random", "random"), ("greedy", "greedy"), ("greedy_undo", "greedy + undo"),
           ("dfs", "DFS"), ("bfs", "BFS")]
RUNS = [("hard45_sonnet_base", "Sonnet base"), ("hard45_sonnet_plan", "Sonnet plan"),
        ("hard45_opus_base", "Opus base"), ("hard45_opus_plan", "Opus plan"),
        ("hard45_codex_base_gpt-6.1-sol", "sol base"), ("hard45_codex_plan_gpt-6.1-sol", "sol plan"),
        ("hard45_codex_base_gpt-6.1-terra", "terra base"), ("hard45_codex_plan_gpt-6.1-terra", "terra plan"),
        ("hard45_codex_base_gpt-6-luna", "luna base"), ("hard45_codex_plan_gpt-6-luna", "luna plan")]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CC = os.path.abspath(os.path.join(ROOT, "..", "cc_runs"))


def ok(r, v, cap):
    return r["win"] and r["verifies"] <= v and (cap is None or r["actions"] <= cap)


def rate(rows, v, cap):
    return 100 * sum(ok(r, v, cap) for r in rows) / len(rows) if rows else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-cap", type=int, default=None)
    ap.add_argument("--max-v", type=int, default=64)
    ap.add_argument("--budgets", default="1,2,3,5,10,20,64")
    ap.add_argument("--runs", default="runs.jsonl", help="baseline file in <game>/analysis_exp2/")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    hard = {g: {l.strip().split("/")[-1] for l in open(f"{CC}/hard45_levels.txt") if l.startswith(g + "/")}
            for g, _ in GAMES}
    base, agent = {}, {}
    for g, _ in GAMES:
        for l in open(f"{ROOT}/{g}/analysis_exp2/{a.runs}"):
            r = json.loads(l)
            if r["level"] in hard[g]:
                base.setdefault((g, r["method"]), []).append(
                    {"win": r["solved"], "verifies": r["verifies"], "actions": r["actions"]})
        for d, name in RUNS:
            last = {}
            for p in glob.glob(f"{CC}/{d}/results_{g}.jsonl"):
                for l in open(p):
                    r = json.loads(l)
                    if r["level"] in hard[g]:
                        last[r["level"]] = r          # a rerun overrides the first try
            agent[(g, name)] = list(last.values())
    cap = a.action_cap
    games = GAMES + [("all", "All 45")]
    def rows_for(src, g, key):
        if g != "all":
            return src.get((g, key), [])
        return [r for gg, _ in GAMES for r in src.get((gg, key), [])]
    out = []
    capnote = f", at most {cap} actions per run" if cap else ""
    for g, gname in games:
        out += [f"### {gname}: success (%) within a verify budget{capnote}", "",
                "| | " + " | ".join(f"≤{b}" for b in budgets) + " |",
                "|---|" + "---|" * len(budgets)]
        for d, name in RUNS:
            rs = rows_for(agent, g, name)
            out.append(f"| {name} | " + " | ".join(f"{rate(rs, b, cap):.0f}" for b in budgets) + " |")
        for m, mname in METHODS:
            rs = rows_for(base, g, m)
            out.append(f"| *{mname}* | " + " | ".join(f"{rate(rs, b, cap):.0f}" for b in budgets) + " |")
        out.append("")
    out += [f"### Verifies a search method needs to match each agent's final success rate{capnote}", "",
            "| Agent | " + " | ".join(f"{gn} (agent %)" for _, gn in games) + " |",
            "|---|" + "---|" * len(games)]
    for d, name in RUNS:
        cells = []
        for g, gname in games:
            target = rate(rows_for(agent, g, name), 10 ** 9, cap)
            best = None
            for m, mname in METHODS:
                rs = rows_for(base, g, m)
                need = next((v for v in range(1, a.max_v + 1) if rate(rs, v, cap) >= target - 1e-9), None)
                if need is not None and (best is None or need < best[0]):
                    best = (need, mname)
            cells.append(f"{best[0]} ({best[1]}) [{target:.0f}%]" if best else f">{a.max_v} [{target:.0f}%]")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
