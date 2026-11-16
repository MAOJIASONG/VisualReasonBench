"""
Stacking trajectories from the two kinds of logs, in one format:

    {"model", "level": "2x3x3_hard_004", "source", "win", "n_actions",
     "actions": [{"n", "tool", "piece", "cells", "ok", "error"}, ...]}

* Claude Code / Codex harness: <run>/private/L??_<level>/meta.json ("log")
* chainbench (pure LLM): <logdir>/level_<i>_<level>/<run>/experiment_log.json
  (when a level has several runs in the given log dirs, the last one is kept)

    python -m chain_analysis.stacking_oracle.collect cc <run_dir> --model NAME --out traj.jsonl
    python -m chain_analysis.stacking_oracle.collect chainbench <logdir>... --model NAME --out traj.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List


def _act(n: int, tool: str, args: Dict, res: Dict) -> Dict:
    cells = args.get("cells")
    ok = (res or {}).get("status") == "success"
    return {"n": n, "tool": tool, "piece": None if args.get("piece_id") is None else str(args.get("piece_id")),
            "cells": [list(map(int, c)) for c in cells] if isinstance(cells, list) and
            all(isinstance(c, (list, tuple)) and len(c) == 3 for c in cells) else None,
            "ok": ok, "error": None if ok else (res or {}).get("error") or (res or {}).get("message")}


def from_cc(run: str, model: str) -> List[Dict]:
    out = []
    for mj in sorted(glob.glob(os.path.join(run, "private", "L*_*", "meta.json"))):
        meta = json.load(open(mj))
        acts = [_act(e["n"], e["action"].get("tool", "invalid"), e["action"], e["result"])
                for e in meta["log"]]
        out.append({"model": model, "level": meta["name"], "source": mj, "win": meta["status"] == "win",
                    "n_actions": meta["n_actions"], "max_actions": meta["max_actions"], "actions": acts})
    return out


def from_chainbench(logdirs: List[str], model: str) -> List[Dict]:
    found: Dict[str, Dict] = {}
    for ld in logdirs:
        for ej in sorted(glob.glob(os.path.join(ld, "level_*", "*", "experiment_log.json"))):
            lvl = re.sub(r"^level_\d+_", "", os.path.basename(os.path.dirname(os.path.dirname(ej))))
            steps = json.load(open(ej))
            acts, n, win = [], 0, False
            for s in steps:
                if s.get("step_type") != "action":
                    continue
                n += 1
                fn = s["tool_call"]["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                acts.append(_act(n, fn["name"], args if isinstance(args, dict) else {}, s.get("tool_result")))
                meta = (s.get("observation") or {}).get("state", {}).get("metadata") or {}
                win = win or bool(meta.get("is_complete"))
            found[lvl] = {"model": model, "level": lvl, "source": ej, "win": win, "n_actions": n,
                          "n_model_steps": max((s.get("step", 0) for s in steps), default=0),
                          "actions": acts}
    return [found[k] for k in sorted(found)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["cc", "chainbench"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = from_cc(a.paths[0], a.model) if a.kind == "cc" else from_chainbench(a.paths, a.model)
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{a.model}: {len(rows)} episodes, {sum(r['win'] for r in rows)} won -> {a.out}")


if __name__ == "__main__":
    main()
