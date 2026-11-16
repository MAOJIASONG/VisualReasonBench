"""Summarize and audit a Puzzle (Luban) run of Claude Code or Codex.

    python summarize_luban.py <run_root> [<run_root> ...]

Per episode: outcome (win / lose = budget used up / playing = agent stopped early), actions,
failed (blocked) actions, pieces removed, and an audit of every tool call: a Read / view_image
must target an obs_*.png in the agent folder and a command must be one `luban-act '<json>'`.
Writes <run_root>/summary.csv and <run_root>/audit.json.
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stacking"))
import summarize_stack as ss  # noqa: E402

OBS_RE = re.compile(r"^(\./)?obs_\d+\.png$")
ACT_RE = re.compile(r"^luban-act '[^']*'\s*$")


def summarize(run):
    ss.OBS_RE, ss.ACT_RE = OBS_RE, ACT_RE          # audit_episode (Claude) reads these globals
    rows, audit = [], {}
    for priv in sorted(glob.glob(os.path.join(run, "private", "L*"))):
        name = os.path.basename(priv)
        mp = os.path.join(priv, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        agent = os.path.join(run, "agent", name)
        is_codex = bool(glob.glob(os.path.join(priv, "rollout_*.jsonl")))
        au = ss.audit_codex(priv, agent, OBS_RE, ACT_RE) if is_codex else ss.audit_episode(priv, agent)
        audit[name] = au
        rows.append({"episode": name, "level": meta["level"], "n_pieces": meta["n_pieces"],
                     "status": meta["status"], "success": int(meta["status"] == "win"),
                     "actions_used": meta["n_actions"], "failed_actions": meta["n_failed"],
                     "pieces_removed": meta.get("n_finished", 0),
                     **{k: v for k, v in au.items() if k != "flags"}, "flagged_calls": len(au["flags"])})
    if not rows:
        print(f"no episodes under {run}/private")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(run, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    json.dump(audit, open(os.path.join(run, "audit.json"), "w"), indent=1)
    acts = sum(r["actions_used"] for r in rows)
    print(f"== {run}")
    print(f"n={len(rows)}  success={sum(r['success'] for r in rows)}  "
          f"stopped_early={sum(r['status'] == 'playing' for r in rows)}  "
          f"budget_out={sum(r['status'] == 'lose' for r in rows)}  avg_actions={acts / len(rows):.1f}  "
          f"failed_action_rate={100 * sum(r['failed_actions'] for r in rows) / max(acts, 1):.1f}%  "
          f"pieces_removed={sum(r['pieces_removed'] for r in rows)}  cost=${sum(r['cost_usd'] for r in rows):.2f}")
    print(f"models: {sorted({m for r in rows for m in str(r['models']).split(',') if m})}  "
          f"api_errors={sum(r['api_errors'] for r in rows)}  flagged_calls={sum(r['flagged_calls'] for r in rows)}")
    for name, au in audit.items():
        for fl in au["flags"]:
            print(f"FLAG {name}: {fl['tool']} {fl['arg'][:300]}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        summarize(os.path.abspath(r))
