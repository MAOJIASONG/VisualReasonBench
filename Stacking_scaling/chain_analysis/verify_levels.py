"""
Check that every generated level is valid.

For each level folder (puzzles_50 / images_batch of the three games):
  - level.json loads and the stored reference plan wins when replayed
    through the game rules;
  - the oracle agrees: N(s0) >= 1 and d(s0) is at most the plan length;
  - the rendered initial frame exists.

    python -m chain_analysis.verify_levels type_3/puzzles_50 type_3_1/puzzles_50 ...
"""

from __future__ import annotations

import glob
import os
import sys

from .envs import load_env, level_dirs


def check(level_dir: str) -> dict:
    env = load_env(level_dir)
    s = env.initial()
    plan = env.reference_plan()
    st = "ok"
    for a in plan:
        s, st = env.step(s, a)
        if st != "ok":
            break
    s0 = env.initial()
    n0, d0 = env.n_solutions(s0), env.dist(s0)
    pngs = glob.glob(os.path.join(level_dir, "*initial*.png"))
    problems = []
    if st != "win":
        problems.append(f"reference plan ends {st}")
    if n0 < 1:
        problems.append("oracle finds no solution")
    if d0 is None or d0 > len(plan):
        problems.append(f"d(s0)={d0} > plan {len(plan)}")
    if not pngs:
        problems.append("no initial frame")
    return {"level": os.path.basename(level_dir), "plan": len(plan),
            "N0": n0, "d0": d0, "problems": problems}


def main(argv=None):
    roots = (argv if argv is not None else sys.argv[1:])
    bad = total = 0
    for root in roots:
        rows = [check(d) for d in level_dirs(root)]
        nb = sum(1 for r in rows if r["problems"])
        total += len(rows)
        bad += nb
        print(f"{root}: {len(rows)} levels, {nb} with problems")
        for r in rows:
            if r["problems"]:
                print(f"  {r['level']}: {'; '.join(r['problems'])}")
    print(f"total {total} levels, {bad} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
