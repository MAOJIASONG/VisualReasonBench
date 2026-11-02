"""
Experiment 2 — search baselines under the same verify budget.

Can the models' scores be reached by trial and error alone?  Non-VLM
baselines play the same levels through the same actions (envs.py) and are
charged the same way a model is:

  action   every executed game action, including the removals of an undo
           and the replay after a reset (irreversible games have no undo)
  verify   every attempt that ends the game: a valve that touches a red cell
           or fills the last tank (FluidNet), a submit (PipeFit), the push
           (Domino).  A failed verify ends the episode; the next attempt
           starts from a reset.

Baselines (all use the same local move generator `env.candidates`, never
the oracle):
  random        uniformly random candidate moves, restart on failure
  greedy        always the top-ranked candidate; later attempts pick rank i
                with probability ~ 2^-i
  greedy_undo   depth-first search in heuristic order with backtracking:
                the greedy line, then the next-best alternative at the
                deepest choice point, ...
  dfs           depth-first search with backtracking in random order
  bfs           breadth-first search over move sequences

Each run gets a large budget; the verify / action counts at the first win
then give the success rate under every smaller budget, so one run per
(level, method, seed) yields the whole curve.

Model results can be added for the comparison:
  --models models.jsonl   lines {"model", "level", "win", "verifies", "actions"}
The report then lists, per method, the fewest verifies at which it matches
each model's success rate.

    python -m chain_analysis.exp2_budget_search --root type_3/puzzles_50 \
        [--models models.jsonl] [--seeds 5] [--max-verifies 64] [--max-actions 2000]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import time
from collections import deque
from typing import Dict, List, Optional

from .envs import ChainEnv, load_env, level_dirs, read_index

METHODS = ["random", "greedy", "greedy_undo", "dfs", "bfs"]
SEEDED = {"random", "greedy", "dfs"}
BUDGETS = [1, 2, 3, 5, 10, 20, 50]


class OutOfBudget(Exception):
    pass


class Solved(Exception):
    pass


class Runner:
    """Executes move sequences on one live game and charges the budget."""

    def __init__(self, env: ChainEnv, max_verifies: int, max_actions: int):
        self.env = env
        self.max_v, self.max_a = max_verifies, max_actions
        self.state = env.initial()
        self.path: List[tuple] = []
        self.actions = self.verifies = self.resets = 0

    def _do(self, a):
        if self.actions >= self.max_a:
            raise OutOfBudget
        self.state, st = self.env.step(self.state, a)
        self.actions += 1
        if st != "ok":
            self.verifies += 1
            if st == "win":
                raise Solved
            if self.verifies >= self.max_v:
                raise OutOfBudget
        return st

    def goto(self, path: List[tuple]):
        """Bring the game to the state after `path`; returns its status."""
        env = self.env
        k = 0
        while k < min(len(path), len(self.path)) and path[k] == self.path[k]:
            k += 1
        tail = self.path[k:]
        if env.status(self.state) or not all(env.undoable(None, a) for a in tail):
            self.state = env.initial()
            self.path, self.resets, k = [], self.resets + 1, 0
        else:
            for a in reversed(tail):
                self._do(env.inverse(a))
            self.path = self.path[:k]
        st = "ok"
        for a in path[k:]:
            self.path.append(a)
            st = self._do(a)
            if st != "ok":
                break
        return st


def _rollouts(env, run: Runner, pick):
    """Episodes of single moves chosen by pick(candidates, attempt)."""
    attempt = 0
    while True:
        path = []
        run.goto(path)
        while True:
            c = env.candidates(run.state)
            if not c:
                break
            path.append(pick(c, attempt))
            if run.goto(path) != "ok":
                break
        attempt += 1
        if not path:
            raise OutOfBudget


def _dfs(env, run: Runner, order):
    seen = set()

    def rec(path):
        st = run.goto(path)
        key = env.config(run.state)
        if st != "ok" or key in seen:
            return
        seen.add(key)
        for a in order(env.candidates(run.state)):
            rec(path + [a])
            if run.goto(path) != "ok":       # back to this node for the next child
                return
    rec([])
    raise OutOfBudget                          # search space exhausted


def _bfs(env, run: Runner):
    seen = {env.config(env.initial())}
    q = deque([[]])
    while q:
        path = q.popleft()
        if run.goto(path) != "ok":
            continue
        for a in env.candidates(run.state):
            nxt, _ = env.step(run.state, a)     # bookkeeping only: which
            key = env.config(nxt)               # configuration a move leads to
            if key not in seen:
                seen.add(key)
                q.append(path + [a])
    raise OutOfBudget


def run_one(env: ChainEnv, method: str, seed: int, max_verifies: int,
            max_actions: int) -> Dict:
    rng = random.Random(seed)
    run = Runner(env, max_verifies, max_actions)
    solved = False
    try:
        if method == "random":
            _rollouts(env, run, lambda c, i: rng.choice(c))
        elif method == "greedy":
            def pick(c, attempt):
                if attempt == 0:
                    return c[0]
                w = [2.0 ** -i for i in range(len(c))]
                return rng.choices(c, weights=w)[0]
            _rollouts(env, run, pick)
        elif method == "greedy_undo":
            _dfs(env, run, lambda c: c)
        elif method == "dfs":
            _dfs(env, run, lambda c: rng.sample(c, len(c)))
        elif method == "bfs":
            _bfs(env, run)
        else:
            raise ValueError(method)
    except Solved:
        solved = True
    except OutOfBudget:
        pass
    return {"solved": solved, "verifies": run.verifies, "actions": run.actions,
            "resets": run.resets}


def _job(args):
    level_dir, method, seed, mv, ma = args
    t = time.time()
    env = load_env(level_dir)
    r = run_one(env, method, seed, mv, ma)
    return {"level": os.path.basename(level_dir), "method": method, "seed": seed,
            **r, "secs": round(time.time() - t, 2)}


def success_at(rows: List[Dict], v: int, a: Optional[int] = None) -> float:
    """Share of runs that win within v verifies (and a actions)."""
    ok = [r["solved"] and r["verifies"] <= v and (a is None or r["actions"] <= a)
          for r in rows]
    return sum(ok) / len(ok) if ok else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-verifies", type=int, default=64)
    ap.add_argument("--max-actions", type=int, default=2000)
    ap.add_argument("--action-cap", type=int, default=60,
                    help="step budget of the model runs, for the second table")
    ap.add_argument("--models", help="JSONL model results to compare against")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)

    root = os.path.normpath(args.root)
    out = args.out or os.path.join(os.path.dirname(root), "analysis_exp2")
    os.makedirs(out, exist_ok=True)
    idx = read_index(root)
    dirs = level_dirs(root)[: args.limit or None]
    methods = args.methods.split(",")
    jobs = [(d, m, s, args.max_verifies, args.max_actions)
            for d in dirs for m in methods
            for s in (range(args.seeds) if m in SEEDED else [0])]
    workers = args.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 4, 120))
    rows = []
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            r["tier"] = idx.get(r["level"], {}).get("tier", "")
            rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs", flush=True)
    with open(os.path.join(out, "runs.jsonl"), "w") as f:
        for r in sorted(rows, key=lambda r: (r["level"], r["method"], r["seed"])):
            f.write(json.dumps(r) + "\n")

    models: Dict[str, List[Dict]] = {}
    if args.models:
        for l in open(args.models):
            if l.strip():
                m = json.loads(l)
                m["level"] = os.path.basename(os.path.normpath(m["level"]))
                models.setdefault(m["model"], []).append(m)

    tiers = [t for t in ("easy", "medium", "hard") if any(r["tier"] == t for r in rows)]
    L = [f"# Exp 2 — search under a verify budget: `{root}`", "",
         f"{len(dirs)} levels; seeded methods averaged over {args.seeds} seeds; "
         f"each run capped at {args.max_verifies} verifies / {args.max_actions} actions.", "",
         "## Success rate (%) by verify budget (no action cap)", "",
         "| method | " + " | ".join(f"≤{b}" for b in BUDGETS) + " | median verifies to win | median actions to win |",
         "|---|" + "---|" * (len(BUDGETS) + 2)]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        won = [r for r in rs if r["solved"]]
        L.append(f"| {m} | " + " | ".join(f"{100 * success_at(rs, b):.0f}" for b in BUDGETS)
                 + f" | {statistics.median(r['verifies'] for r in won) if won else '–'}"
                 + f" | {statistics.median(r['actions'] for r in won) if won else '–'} |")
    for name, ms in models.items():
        L.append(f"| **{name}** | " + " | ".join(
            f"{100 * success_at([{'solved': x['win'], 'verifies': x.get('verifies', 1), 'actions': x.get('actions', 0)} for x in ms], b):.0f}"
            for b in BUDGETS) + " | | |")
    L += ["", f"## Success rate (%) by verify budget, at most {args.action_cap} actions", "",
          "| method | " + " | ".join(f"≤{b}" for b in BUDGETS) + " |",
          "|---|" + "---|" * len(BUDGETS)]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        L.append(f"| {m} | " + " | ".join(
            f"{100 * success_at(rs, b, args.action_cap):.0f}" for b in BUDGETS) + " |")
    if tiers:
        L += ["", "## By tier (≤5 verifies / unlimited within the run cap)", "",
              "| method | " + " | ".join(f"{t} ≤5 | {t} all" for t in tiers) + " |",
              "|---|" + "---|---|" * len(tiers)]
        for m in methods:
            cells = []
            for t in tiers:
                rs = [r for r in rows if r["method"] == m and r["tier"] == t]
                cells += [f"{100 * success_at(rs, 5):.0f}",
                          f"{100 * success_at(rs, args.max_verifies):.0f}"]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
    if models:
        L += ["", "## Verifies a method needs to match each model", "",
              "| model | model success % | " + " | ".join(methods) + " |",
              "|---|---|" + "---|" * len(methods)]
        for name, ms in models.items():
            lv = {x["level"] for x in ms}
            target = sum(x["win"] for x in ms) / len(ms)
            cells = []
            for m in methods:
                rs = [r for r in rows if r["method"] == m and r["level"] in lv]
                need = next((v for v in range(1, args.max_verifies + 1)
                             if success_at(rs, v) >= target), None)
                cells.append(str(need) if need else f">{args.max_verifies}")
            L.append(f"| {name} | {100 * target:.0f} | " + " | ".join(cells) + " |")
    L += ["", "Reading: if a search method matches a model at the same verify budget, "
          "the model's score is reachable by trial and error.  If cutting the "
          "budget hurts backtracking search much more than the model, the model "
          "is planning; if both fall together, the model also relies on retries.", ""]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
