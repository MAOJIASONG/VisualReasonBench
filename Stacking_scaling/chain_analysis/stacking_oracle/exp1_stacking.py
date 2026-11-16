"""
Experiment 1 on Stacking: dead ends and step value.

Stacking is reversible (`pickup` takes any piece back), so a dead end is not
absorbing: it is a *dead state* — the placed pieces cannot all stay where they
are in any completion — and leaving it takes one or more pickups.

Per level (oracle, `levels.jsonl`):
  legal first moves, how many of them are traps (lead to a dead state), how
  many of those traps are hidden (not exposed: every remaining piece still fits
  and every free cell can still be covered), and the same along the stored
  solution (traps per step, earliest step with a trap).
Per trajectory (`steps.jsonl` / `episodes.jsonl`):
  fatal step = first accepted placement that enters a dead state; exposure =
  first later action where the dead state becomes visible; repair = first
  action back in a solvable state; consequence delay = exposure - fatal;
  share of actions spent in dead states; whether the episode ended in one.
Action numbers count every logged action (rejected ones and `state` calls
too), as the action budget does.

    python -m chain_analysis.stacking_oracle.exp1_stacking --traj traj.jsonl... --out DIR [--levels-only]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stx
from collections import defaultdict
from typing import Dict, List, Optional

from .core import all_level_names, bits, level

DIFF = ("easy", "mid", "hard")


# ---------------------------------------------------------------------------
# Level statistics
# ---------------------------------------------------------------------------

def trap_counts(L, occ: int, un: int) -> Dict:
    legal = L.legal_moves(occ, un)
    traps = hidden = 0
    for p, m in legal:
        o2, u2 = occ | m, un & ~(1 << p)
        if not L.solvable(o2, u2):
            traps += 1
            hidden += not L.exposed(o2, u2)
    return {"legal": len(legal), "traps": traps, "hidden": hidden}


def level_stats(name: str) -> Dict:
    L = level(name)
    occ, un, per_step = 0, L.all_pieces, []
    for k, (p, m) in enumerate(L.solution_order()):
        per_step.append({"step": k + 1, **trap_counts(L, occ, un)})
        occ |= m
        un &= ~(1 << p)
    first = per_step[0]
    earliest = next((s["step"] for s in per_step if s["traps"]), None)
    return {"level": name, "difficulty": name.split("_")[1], "box": list(L.box), "pieces": L.n,
            "legal_first": first["legal"], "traps_first": first["traps"], "hidden_first": first["hidden"],
            "trap_share_first": first["traps"] / first["legal"] if first["legal"] else 0.0,
            "traps_per_step": stx.mean(s["traps"] for s in per_step),
            "trap_share_per_step": stx.mean(s["traps"] / s["legal"] for s in per_step if s["legal"]),
            "hidden_per_step": stx.mean(s["hidden"] for s in per_step),
            "earliest_trap_step": earliest, "per_step": per_step}


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def analyse(ep: Dict) -> Dict:
    L = level(ep["level"])
    occ, un = 0, L.all_pieces
    where: Dict[int, int] = {}                       # piece -> mask
    solv, steps, mismatch = True, [], 0
    placed_n: Dict[int, int] = {}
    wasted_places = 0
    for a in ep["actions"]:
        changed = False
        if a["tool"] == "place" and a["ok"] and a["piece"] is not None and a["cells"]:
            p = int(a["piece"])
            m, err = L.check_place(occ, un, p, [tuple(c) for c in a["cells"]])
            if m is None:
                mismatch += 1
                m = L.mask_of([tuple(c) for c in a["cells"]])
            if m is not None and (un >> p) & 1:
                occ |= m; un &= ~(1 << p); where[p] = m; placed_n[p] = a["n"]; changed = True
        elif a["tool"] == "pickup" and a["ok"] and a["piece"] is not None:
            p = int(a["piece"])
            if p in where:
                occ &= ~where.pop(p); un |= 1 << p; wasted_places += 1; changed = True
        elif a["tool"] == "place" and not a["ok"] and a["piece"] is not None and a["cells"]:
            m, err = L.check_place(occ, un, int(a["piece"]), [tuple(c) for c in a["cells"]])
            mismatch += m is not None                # the game rejected what we accept
        if changed:
            solv = L.solvable(occ, un)
        steps.append({"n": a["n"], "tool": a["tool"], "ok": a["ok"], "changed": changed,
                      "solvable": solv, "exposed": (not solv) and L.exposed(occ, un),
                      "placed": bin(L.all_pieces & ~un).count("1")})
    n_end = steps[-1]["n"] if steps else 0
    # dead stretches
    stretches, cur = [], None
    for s in steps:
        if cur is None and not s["solvable"]:
            cur = {"fatal": s["n"], "exposed": None, "repair": None}
        if cur is not None:
            if s["solvable"]:
                cur["repair"] = s["n"]; stretches.append(cur); cur = None
            elif cur["exposed"] is None and s["exposed"]:
                cur["exposed"] = s["n"]
    if cur is not None:
        stretches.append(cur)
    dead_actions = sum(1 for s in steps if not s["solvable"])
    first = stretches[0] if stretches else None
    return {"model": ep["model"], "level": ep["level"], "difficulty": ep["level"].split("_")[1],
            "win": ep["win"], "n_actions": ep["n_actions"],
            "rejected": sum(1 for a in ep["actions"] if not a["ok"]),
            "places": sum(1 for s in steps if s["tool"] == "place" and s["changed"]),
            "pickups": sum(1 for s in steps if s["tool"] == "pickup" and s["changed"]),
            "wasted_places": wasted_places, "dead_entries": len(stretches),
            "fatal_step": first["fatal"] if first else None,
            "fatal_placed_before": next((s["placed"] - 1 for s in steps if first and s["n"] == first["fatal"]), None),
            "exposure_step": first["exposed"] if first else None,
            "delay": (first["exposed"] - first["fatal"]) if first and first["exposed"] else None,
            "repair_step": first["repair"] if first else None,
            "repair_lag": (first["repair"] - first["fatal"]) if first and first["repair"] else None,
            "dead_share": dead_actions / len(steps) if steps else 0.0,
            "ended_dead": bool(steps) and not steps[-1]["solvable"],
            "ended_dead_hidden": bool(steps) and not steps[-1]["solvable"] and not steps[-1]["exposed"],
            "engine_mismatch": mismatch, "stretches": stretches, "n_end": n_end}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x, n):
    return f"{100 * x / n:.0f}%" if n else "–"


def _med(xs):
    xs = [x for x in xs if x is not None]
    return f"{stx.median(xs):.1f}" if xs else "–"


def report(levels: List[Dict], eps: List[Dict]) -> str:
    out = ["# Experiment 1 on Stacking: dead states and step value", "",
           "Stacking is reversible (`pickup`), so a wrong placement leads to a *dead state* "
           "(the placed pieces fit no completion) that can be left again with pickups. "
           "Oracle: exact cover of the free cells by the unplaced pieces "
           "(`stacking_oracle/core.py`); a dead state is *hidden* while every remaining piece "
           "still fits somewhere and every free cell can still be covered.", ""]
    if levels:
        out += ["## Traps per level (oracle)", "",
                "| difficulty | levels | pieces | legal first moves | traps among them | hidden traps "
                "| trap share, first move | trap share per step | earliest trap step |",
                "|---|---|---|---|---|---|---|---|---|"]
        for d in DIFF + ("all",):
            ls = [l for l in levels if d == "all" or l["difficulty"] == d]
            if not ls:
                continue
            out.append(f"| {d} | {len(ls)} | {stx.mean(l['pieces'] for l in ls):.1f} | "
                       f"{stx.mean(l['legal_first'] for l in ls):.0f} | {stx.mean(l['traps_first'] for l in ls):.0f} | "
                       f"{stx.mean(l['hidden_first'] for l in ls):.0f} | "
                       f"{100 * stx.mean(l['trap_share_first'] for l in ls):.0f}% | "
                       f"{100 * stx.mean(l['trap_share_per_step'] for l in ls):.0f}% | "
                       f"{_med([l['earliest_trap_step'] for l in ls])} (median) |")
        out.append("")
    if eps:
        models = sorted({e["model"] for e in eps})
        out += ["## Trajectories", "",
                "fatal = first accepted placement into a dead state; delay = actions from it until the "
                "dead state is visible; repaired = left the dead state again (pickups); "
                "failed episodes split by how they ended.", "",
                "| model | episodes | won | with a fatal step | fatal step (median) | delay to exposure (median) "
                "| never exposed | repaired | actions in dead states | failed: ended dead | failed: ended solvable "
                "| rejected actions |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            es = [e for e in eps if e["model"] == m]
            f = [e for e in es if e["fatal_step"] is not None]
            lost = [e for e in es if not e["win"]]
            out.append(
                f"| {m} | {len(es)} | {_pct(sum(e['win'] for e in es), len(es))} | {_pct(len(f), len(es))} | "
                f"{_med([e['fatal_step'] for e in f])} | {_med([e['delay'] for e in f])} | "
                f"{_pct(sum(e['exposure_step'] is None and e['repair_step'] is None for e in f), len(f))} | "
                f"{_pct(sum(e['repair_step'] is not None for e in f), len(f))} | "
                f"{100 * stx.mean(e['dead_share'] for e in es):.0f}% | "
                f"{_pct(sum(e['ended_dead'] for e in lost), len(lost))} | "
                f"{_pct(sum(not e['ended_dead'] for e in lost), len(lost))} | "
                f"{100 * sum(e['rejected'] for e in es) / max(1, sum(e['n_actions'] for e in es)):.0f}% |")
        out += ["", "### By difficulty (won / with a fatal step / ended dead)", "",
                "| model | " + " | ".join(DIFF) + " |", "|---|" + "---|" * len(DIFF)]
        for m in models:
            cells = []
            for d in DIFF:
                es = [e for e in eps if e["model"] == m and e["difficulty"] == d]
                cells.append(f"{sum(e['win'] for e in es)}/{len(es)} · {sum(e['fatal_step'] is not None for e in es)}"
                             f" · {sum(e['ended_dead'] for e in es)}" if es else "–")
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        mism = sum(e["engine_mismatch"] for e in eps)
        out += ["", f"Engine check: {mism} logged actions where the oracle's rules disagree with the game's "
                    "verdict (0 expected)."]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels-only", action="store_true")
    ap.add_argument("--no-levels", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    levels = []
    lpath = os.path.join(a.out, "levels.jsonl")
    if not a.no_levels:
        if os.path.exists(lpath):
            levels = [json.loads(l) for l in open(lpath)]
        else:
            for n in all_level_names():
                levels.append(level_stats(n))
                print(f"level {n}: {levels[-1]['traps_first']}/{levels[-1]['legal_first']} traps", flush=True)
            with open(lpath, "w") as f:
                for l in levels:
                    f.write(json.dumps(l) + "\n")
    eps = []
    if not a.levels_only:
        for t in a.traj:
            for line in open(t):
                eps.append(analyse(json.loads(line)))
        with open(os.path.join(a.out, "episodes.jsonl"), "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
    open(os.path.join(a.out, "REPORT.md"), "w").write(report(levels, eps))
    print(f"-> {a.out}/REPORT.md")


if __name__ == "__main__":
    main()
