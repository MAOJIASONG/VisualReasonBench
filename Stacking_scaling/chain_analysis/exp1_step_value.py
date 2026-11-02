"""
Experiment 1 — dead ends and step value.

For every step of a trajectory the oracle (envs.py) gives
  N(s)   distinct winning end states still reachable (capped)
  d(s)   fewest actions to win, counting removals a wrong piece needs;
         an irreversible dead end costs a restart: d = d(s0) + 1

Per step:
  V      = N(s') / N(s)      solution retention (0 = walked into a dead end)
  V_rand = mean V over the legal progress actions in s (open / place / push)
  V_best = max  V over the same actions
  dd     = d(s) - d(s')      +1 = one step closer, < 0 = moved away

Per trajectory:
  fatal_step        first step with N(s) > 0 and N(s') = 0
  exposed_step      first step at or after it where the game says so (a
                    failed verify: red cell touched, failed submit, losing push)
  consequence_delay exposed_step - fatal_step (None if never exposed;
                    `censored` then counts the steps it stayed hidden)
  repaired_step     first later step with N > 0 again (undo / remove)
  first_neg_dd      first step with dd < 0
  wasted            number of steps with dd <= 0

Per level (from the reference plan):
  traps_at_start    progress actions in s0 that lead into a dead end
  earliest_trap     first plan step (1-based) where a dead-end move exists
  trap_steps        plan steps that offer at least one dead-end move
  mean_traps        mean dead-end moves per plan state

Trajectories are JSONL, one episode list per line:
  {"model": "...", "level": "type_3/puzzles_50/P01_.../", "actions": [[...], ...]}
Action lists use the forms in envs.py; ["reset"] starts over inside a line.
Without --traj, the demo trajectories are the reference plan and a random
rollout of the candidate-move policy per level (exp2's random baseline).

    python -m chain_analysis.exp1_step_value --root type_3/puzzles_50 \
        [--traj runs.jsonl] [--out out_dir]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from typing import Dict, List, Optional

from .envs import ChainEnv, load_env, level_dirs, read_index, to_action

RAND_SAMPLE = 400          # V_rand averages at most this many sampled actions


def _ratio(a: int, b: int) -> Optional[float]:
    return None if b == 0 else a / b


def step_values(env: ChainEnv, s, rng: random.Random) -> Dict:
    """V_rand / V_best / dead-end count over the progress actions in s."""
    acts = [a for a in env.legal(s) if env.is_progress(a)]
    n = env.n_solutions(s)
    if not acts or n == 0:
        return {"n_actions": len(acts), "V_rand": None, "V_best": None, "n_dead": None}
    ns = env.n_after(s, acts)
    sample = ns if len(ns) <= RAND_SAMPLE else rng.sample(ns, RAND_SAMPLE)
    return {"n_actions": len(acts),
            "V_rand": round(statistics.mean(min(x, n) / n for x in sample), 4),
            "V_best": round(max(min(x, n) for x in ns) / n, 4),
            "n_dead": sum(1 for x in ns if x == 0)}


def analyze_trajectory(env: ChainEnv, actions: List, rng: random.Random,
                       with_baselines: bool = True) -> Dict:
    s0 = env.initial()
    d0 = env.dist(s0)
    restart = (d0 or 0) + 1

    def d_of(s):
        d = env.dist(s)
        return restart if d is None else d

    s = s0
    steps = []
    fatal = exposed = repaired = first_neg = None
    for t, a in enumerate(actions, 1):
        a = to_action(a)
        n, d = env.n_solutions(s), d_of(s)
        base = step_values(env, s, rng) if with_baselines and a[0] != "reset" else {}
        try:
            s2, status = env.step(s, a)
        except ValueError as e:                       # illegal: no state change
            steps.append({"t": t, "action": a, "illegal": str(e)})
            continue
        n2, d2 = env.n_solutions(s2), d_of(s2)
        dd = d - d2
        rec = {"t": t, "action": a, "status": status, "N": n, "N_next": n2,
               "V": None if n == 0 else round(min(n2, n) / n, 4),
               "d": d, "d_next": d2, "dd": dd, **base}
        steps.append(rec)
        if fatal is None and n > 0 and n2 == 0 and a[0] != "reset":
            fatal = t
        if fatal is not None and exposed is None and repaired is None and status == "lose":
            exposed = t
        if fatal is not None and repaired is None and exposed is None and t > fatal and n2 > 0:
            repaired = t
        if first_neg is None and dd < 0 and a[0] != "reset":
            first_neg = t
        s = s2
    moves = [r for r in steps if "V" in r and r["action"][0] != "reset"]
    vs = [r["V"] for r in moves if r["V"] is not None]
    gain = [r["V"] - r["V_rand"] for r in moves if r.get("V_rand") is not None and r["V"] is not None]
    return {
        "n_steps": len(actions),
        "win": any(r.get("status") == "win" for r in steps),
        "fatal_step": fatal,
        "exposed_step": exposed,
        "consequence_delay": None if fatal is None or exposed is None else exposed - fatal,
        "censored": (len(actions) - fatal) if fatal is not None and exposed is None
                    and repaired is None else None,
        "repaired_step": repaired,
        "first_neg_dd": first_neg,
        "wasted": sum(1 for r in moves if r["dd"] <= 0),
        "mean_V": round(statistics.mean(vs), 4) if vs else None,
        "mean_V_minus_rand": round(statistics.mean(gain), 4) if gain else None,
        "illegal": sum(1 for r in steps if "illegal" in r),
        # a won game must have left every dead end; if not, the oracle
        # missed a solution (N(s) is a lower bound) -- reported, not hidden
        "oracle_miss": bool(fatal is not None and repaired is None and exposed is None
                            and any(r.get("status") == "win" for r in steps)),
        "steps": steps,
    }


def level_traps(env: ChainEnv) -> Dict:
    """Dead-end moves offered along the reference plan."""
    s = env.initial()
    per_state = []
    for a in env.reference_plan():
        acts = [b for b in env.legal(s) if env.is_progress(b)]
        ns = env.n_after(s, acts) if acts else []
        per_state.append(sum(1 for x in ns if x == 0))
        s, st = env.step(s, a)
        if st != "ok":
            break
    first = next((i + 1 for i, k in enumerate(per_state) if k > 0), None)
    return {"plan_len": len(env.reference_plan()),
            "N0": env.n_solutions(env.initial()),
            "d0": env.dist(env.initial()),
            "traps_at_start": per_state[0] if per_state else 0,
            "earliest_trap": first,
            "trap_steps": sum(1 for k in per_state if k > 0),
            "mean_traps": round(statistics.mean(per_state), 2) if per_state else 0}


def random_rollout(env: ChainEnv, rng: random.Random, max_steps: int = 60) -> List:
    """One episode of uniformly random candidate moves (until win / lose)."""
    s, acts = env.initial(), []
    for _ in range(max_steps):
        c = env.candidates(s)
        if not c:
            break
        a = rng.choice(c)
        acts.append(list(a))
        s, st = env.step(s, a)
        if st != "ok":
            break
    return acts


def _fmt(x) -> str:
    if x is None:
        return "–"
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="puzzles_50 or images_batch dir")
    ap.add_argument("--traj", help="JSONL trajectories (default: demo rollouts)")
    ap.add_argument("--out", help="output dir (default: <root>/../analysis_exp1)")
    ap.add_argument("--cap", type=int, default=500, help="N(s) cap")
    ap.add_argument("--limit", type=int, default=0, help="first N levels only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    root = os.path.normpath(args.root)
    out = args.out or os.path.join(os.path.dirname(root), "analysis_exp1")
    os.makedirs(out, exist_ok=True)
    idx = read_index(root)
    dirs = level_dirs(root)[: args.limit or None]
    rng = random.Random(args.seed)
    envs: Dict[str, ChainEnv] = {}

    def env_of(path):
        key = os.path.basename(os.path.normpath(path))
        if key not in envs:
            envs[key] = load_env(os.path.join(root, key), cap=args.cap)
        return key, envs[key]

    # per-level trap table
    lv_rows = []
    for d in dirs:
        key, env = env_of(d)
        r = {"level": key, "tier": idx.get(key, {}).get("tier", ""), **level_traps(env)}
        lv_rows.append(r)
        print(f"  {key}: {r}", flush=True)

    # trajectories
    if args.traj:
        trajs = [json.loads(l) for l in open(args.traj) if l.strip()]
    else:
        trajs = []
        for d in dirs:
            key, env = env_of(d)
            trajs.append({"model": "reference", "level": key,
                          "actions": [list(a) for a in env.reference_plan()]})
            trajs.append({"model": "random", "level": key,
                          "actions": random_rollout(env, rng)})
    tr_rows = []
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for tr in trajs:
            key, env = env_of(tr["level"])
            res = analyze_trajectory(env, tr["actions"], rng)
            steps = res.pop("steps")
            row = {"model": tr.get("model", ""), "level": key,
                   "tier": idx.get(key, {}).get("tier", ""), **res}
            tr_rows.append(row)
            f.write(json.dumps({**row, "steps": steps}, default=list) + "\n")
    json.dump({"levels": lv_rows, "trajectories": tr_rows},
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=list)

    # markdown report
    L = [f"# Exp 1 — dead ends and step value: `{root}`", "",
         "## Traps per level (along the reference plan)", "",
         "| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |",
         "|---|---|---|---|---|---|---|---|"]
    for tier in ("easy", "medium", "hard", ""):
        rs = [r for r in lv_rows if r["tier"] == tier]
        if rs:
            L.append(f"| {tier or '–'} | {len(rs)} | {_mean(r['plan_len'] for r in rs)} | "
                     f"{_mean(r['N0'] for r in rs)} | {_mean(r['traps_at_start'] for r in rs)} | "
                     f"{_mean(r['earliest_trap'] for r in rs)} | {_mean(r['trap_steps'] for r in rs)} | "
                     f"{_mean(r['mean_traps'] for r in rs)} |")
    L += ["", "## Trajectories", "",
          "| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in sorted({r["model"] for r in tr_rows}):
        for tier in ("easy", "medium", "hard", ""):
            rs = [r for r in tr_rows if r["model"] == model and r["tier"] == tier]
            if not rs:
                continue
            fat = [r for r in rs if r["fatal_step"] is not None]
            L.append(
                f"| {model} | {tier or '–'} | {len(rs)} | "
                f"{100 * sum(r['win'] for r in rs) / len(rs):.0f} | "
                f"{100 * len(fat) / len(rs):.0f} | {_fmt(_mean(r['fatal_step'] for r in fat))} | "
                f"{_fmt(_mean(r['consequence_delay'] for r in fat))} | "
                f"{sum(1 for r in fat if r['censored'] is not None)} | "
                f"{_fmt(100 * sum(1 for r in fat if r['repaired_step']) / len(fat) if fat else None)} | "
                f"{_fmt(_mean(r['mean_V'] for r in rs))} | {_fmt(_mean(r['mean_V_minus_rand'] for r in rs))} | "
                f"{_fmt(_mean(r['wasted'] for r in rs))} |")
    L += ["", "Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* "
          "first such step; *delay* steps until the game exposes it (failed verify); "
          "*hidden at end* fatal runs whose dead end was never exposed or repaired; "
          "*V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the "
          "same state; *wasted* steps with d(s') ≥ d(s).", "",
          f"Oracle check: {sum(r['oracle_miss'] for r in tr_rows)} won trajectories "
          "still marked as stuck in a dead end (0 = the oracle saw every winning line).", ""]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
