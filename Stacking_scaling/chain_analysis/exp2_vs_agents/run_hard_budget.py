"""Rerun the Experiment-2 search baselines on the 45 hard levels with a large budget, so that
the verify count each method needs to reach the agents' scores can be read off (the first run
stopped at 64 verifies / 2000 actions).

    python -m chain_analysis.exp2_vs_agents.run_hard_budget --max-verifies 1000 --max-actions 100000
Writes <game>/analysis_exp2/runs_hard_v<max-verifies>.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time

from chain_analysis.exp2_budget_search import METHODS, SEEDED, _job

GAMES = ["type_3", "type_3_1", "type_2_3"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=os.path.join(ROOT, "..", "cc_runs", "hard45_levels.txt"))
    ap.add_argument("--max-verifies", type=int, default=1000)
    ap.add_argument("--max-actions", type=int, default=100000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    levels = [l.strip() for l in open(a.levels) if l.strip()]
    jobs = [(os.path.join(ROOT, l), m, s, a.max_verifies, a.max_actions)
            for l in levels for m in METHODS for s in (range(a.seeds) if m in SEEDED else [0])]
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(jobs)} runs on {workers} workers, budget {a.max_verifies} verifies / "
          f"{a.max_actions} actions", flush=True)
    rows = {g: [] for g in GAMES}
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (job, r) in enumerate(zip(jobs, pool.imap(_job, jobs)), 1):
            game = os.path.relpath(job[0], ROOT).split(os.sep)[0]
            r["tier"] = "hard"
            rows[game].append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)
    for g, rs in rows.items():
        p = os.path.join(ROOT, g, "analysis_exp2", f"runs_hard_v{a.max_verifies}.jsonl")
        with open(p, "w") as f:
            for r in sorted(rs, key=lambda r: (r["level"], r["method"], r["seed"])):
                f.write(json.dumps(r) + "\n")
        print("->", p)


if __name__ == "__main__":
    main()
