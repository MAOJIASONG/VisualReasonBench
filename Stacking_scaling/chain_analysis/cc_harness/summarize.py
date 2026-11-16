"""Summarize and audit a Claude Code run on the CHAIN games.

    python summarize.py <run_root> --model NAME [--export]

For every episode in <run_root>/private/<game>__<level>/:
  * game outcome from the session state (won / lost / playing, actions, failed attempts)
  * Claude usage summed over all launches (turns, tokens, cost reported by Claude Code)
  * audit of every tool call: Read must target an obs_*.png in the agent folder, Bash must
    be a single `chain-act '<json>'` call; anything else is flagged
Writes <run_root>/summary.csv and <run_root>/audit.json.
With --export, also writes traj_<game>.jsonl / results_<game>.jsonl (Experiment 1/2 input).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stacking"))
from summarize_stack import audit_codex  # noqa: E402  (Codex code-mode audit, shared)

SS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OBS_RE = re.compile(r"^(\./)?obs_\d+_[a-z]+\.png$")
ACT_RE = re.compile(r"^chain-act '[^']*'\s*$")


def load_events(path):
    for line in open(path):
        try:
            yield json.loads(line)
        except Exception:
            continue


def audit_episode(priv, agent):
    calls, flags, denials = 0, [], 0
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0}
    cost, turns, dur, launches, errors = 0.0, 0, 0, 0, 0
    for tr in sorted(glob.glob(os.path.join(priv, "transcript_*.jsonl"))):
        launches += 1
        for ev in load_events(tr):
            if ev.get("type") == "assistant":
                for blk in ev.get("message", {}).get("content", []):
                    if blk.get("type") != "tool_use":
                        continue
                    calls += 1
                    name, inp = blk.get("name"), blk.get("input", {})
                    if name == "Read":
                        fp = inp.get("file_path", "")
                        rel = os.path.relpath(fp, agent) if os.path.isabs(fp) else fp
                        if not OBS_RE.match(rel):
                            flags.append({"tool": "Read", "arg": fp})
                    elif name == "Bash":
                        cmd = inp.get("command", "").strip()
                        if not ACT_RE.match(cmd):
                            flags.append({"tool": "Bash", "arg": cmd})
                    else:
                        flags.append({"tool": name, "arg": json.dumps(inp)[:200]})
            elif ev.get("type") == "result":
                cost += float(ev.get("total_cost_usd") or 0)
                turns += int(ev.get("num_turns") or 0)
                dur += int(ev.get("duration_ms") or 0)
                errors += 1 if ev.get("is_error") else 0
                denials += len(ev.get("permission_denials") or [])
                for k in usage:
                    usage[k] += int((ev.get("usage") or {}).get(k) or 0)
    return {"tool_calls": calls, "flags": flags, "permission_denials": denials,
            "launches": launches, "api_errors": errors, "claude_turns": turns,
            "cost_usd": round(cost, 4), "duration_min": round(dur / 60000, 1), **usage}


def game_state(priv):
    out = subprocess.run([sys.executable, "-m", "chain_analysis.play_cli", "status", priv],
                         cwd=SS, capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--model", required=True)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    run = os.path.abspath(a.run_root)

    rows, audit = [], {}
    for priv in sorted(glob.glob(os.path.join(run, "private", "*__*"))):
        name = os.path.basename(priv)
        agent = os.path.join(run, "agent", name)
        st = game_state(priv)
        is_codex = bool(glob.glob(os.path.join(priv, "rollout_*.jsonl")))
        au = (audit_codex(priv, agent, OBS_RE, ACT_RE) if is_codex
              else audit_episode(priv, agent))
        audit[name] = au
        rows.append({"episode": name, "game": name.split("__")[0],
                     "status": st.get("status"), "actions_used": st.get("actions_used"),
                     "failed_attempts": st.get("failed_attempts"),
                     **{k: v for k, v in au.items() if k != "flags"},
                     "flagged_calls": len(au["flags"])})

    if not rows:
        sys.exit(f"no episodes under {run}/private")
    with open(os.path.join(run, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                           restval="")
        w.writeheader()
        w.writerows(rows)
    json.dump(audit, open(os.path.join(run, "audit.json"), "w"), indent=1)

    cols = ["episode", "status", "actions_used", "failed_attempts", "claude_turns",
            "output_tokens", "cost_usd", "duration_min", "flagged_calls", "permission_denials"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    for name, au in audit.items():
        for fl in au["flags"]:
            print(f"FLAG {name}: {fl['tool']} {fl['arg']}")

    if a.export:
        for game in sorted({r["game"] for r in rows}):
            sess = sorted(glob.glob(os.path.join(run, "private", f"{game}__*")))
            subprocess.run([sys.executable, "-m", "chain_analysis.play_cli", "export", *sess,
                            "--model", a.model,
                            "--out", os.path.join(run, f"traj_{game}.jsonl"),
                            "--results", os.path.join(run, f"results_{game}.jsonl")],
                           cwd=SS, check=False)
        print(f"exported traj_<game>.jsonl / results_<game>.jsonl to {run}")


if __name__ == "__main__":
    main()
