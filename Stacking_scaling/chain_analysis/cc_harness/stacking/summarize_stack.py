"""Summarize and audit a Claude Code run on the Stacking levels.

    python summarize_stack.py <run_root> [<run_root> ...]

Per episode: outcome (win / lose / playing = agent stopped early), actions used, failed
actions, Claude usage, and an audit of every tool call (Read must target an obs_*.png in
the agent folder, Bash must be one `stack-act '<json>'` call). Writes <run_root>/summary.csv
and <run_root>/audit.json, and prints success by difficulty.
"""
import csv
import glob
import json
import os
import re
import shlex
import sys

OBS_RE = re.compile(r"^(\./)?obs_\d+\.png$")
ACT_RE = re.compile(r"^stack-act '[^']*'\s*$")


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
    cost, turns, dur, launches, errors, models = 0.0, 0, 0, 0, 0, set()
    for tr in sorted(glob.glob(os.path.join(priv, "transcript_*.jsonl"))):
        launches += 1
        for ev in load_events(tr):
            if ev.get("type") == "assistant":
                m = ev.get("message", {})
                if m.get("model"):
                    models.add(m["model"])
                for blk in m.get("content", []):
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
            "cost_usd": round(cost, 4), "duration_min": round(dur / 60000, 1),
            "models": ",".join(sorted(models)), **usage}


# ---- Codex: every tool use goes through a JavaScript `exec` call (code mode). An exec call
# passes the audit only if it is a single view_image of an obs_*.png in the agent folder or a
# single exec_command running one stack-act; any other code or tool is flagged.
JS_CALL_RE = re.compile(r"tools\.(\w+)\(")
# An optional `detail` argument only picks the image resolution (e.g. "original"), so it is allowed.
JS_VIEW_RE = re.compile(r"""tools\.view_image\(\s*\{\s*"?path"?\s*:\s*["']([^"']+)["']"""
                        r"""(?:\s*,\s*"?detail"?\s*:\s*["'][a-z]+["'])?\s*\}\s*\)""")
JS_CMD_RE = re.compile(r"""tools\.exec_command\(\s*\{\s*"?cmd"?\s*:\s*("(?:[^"\\]|\\.)*"|`[^`$\\]*`)""")
JS_BAD_RE = re.compile(r"\b(for|while|if|function|return|let|var|new|Math|JSON|Array|Object|eval)\b|=>|\+\+|\*")


def agent_rel(path, agent):
    """Path relative to the agent folder. An absolute path is matched on its
    .../agent/<episode>/ part, so a run copied to another machine audits the same."""
    if not os.path.isabs(path):
        return path
    key = "/agent/" + os.path.basename(agent.rstrip("/")) + "/"
    return path.split(key, 1)[1] if key in path else os.path.relpath(path, agent)


def act_ok(cmd, act_re):
    """One call of the action program with one argument and nothing else. Besides the
    canonical form (act_re), any shell quoting of the single argument is accepted, e.g.
    chain-act "[\\"place\\", ...]"; a second command, pipe, redirect, etc. is not."""
    if act_re.match(cmd):
        return True
    prog = re.match(r"\^(\S+)", act_re.pattern).group(1)
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return False
    return len(toks) == 2 and toks[0] == prog and not re.search(r"[$`]", cmd)


def audit_exec_js(code, agent, obs_re=OBS_RE, act_re=ACT_RE):
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("// @exec"))
    tools = JS_CALL_RE.findall(code)
    if len(tools) != 1:
        return f"{len(tools)} tool uses in one exec"
    if tools[0] == "view_image":
        m = JS_VIEW_RE.search(code)
        if not m:
            return "unparsed view_image"
        rel = agent_rel(m.group(1), agent)
        if not obs_re.match(rel):
            return f"view_image of {m.group(1)}"
        rest = code.replace(m.group(0), "")
    elif tools[0] == "write_stdin":
        # Polling a still-running command for its output (empty input) is allowed; typing
        # anything into it is not.
        if not re.search(r"""\bchars"?\s*:\s*(""|'')""", code):
            return "write_stdin with input"
        rest = re.sub(r"""tools\.write_stdin\(\s*\{[^}]*\}\s*\)""", "", code)
    elif tools[0] == "exec_command":
        m = JS_CMD_RE.search(code)
        if not m:
            return "unparsed exec_command"
        lit = m.group(1)
        cmd = (lit[1:-1] if lit.startswith("`") else json.loads(lit)).strip()
        if not act_ok(cmd, act_re):
            return f"command {cmd[:200]}"
        rest = code.replace(m.group(1), '""')
    else:
        return f"tool {tools[0]}"
    rest = rest.replace("JSON.stringify(", "(")   # printing the tool result as JSON is harmless
    if JS_BAD_RE.search(rest):
        return "extra code"
    return None


def audit_codex(priv, agent, obs_re=OBS_RE, act_re=ACT_RE):
    calls, flags, views, errors, launches, models = 0, [], 0, 0, 0, set()
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
             "reasoning_output_tokens": 0}
    for ro in sorted(glob.glob(os.path.join(priv, "rollout_*.jsonl"))):
        for ev in load_events(ro):
            p = ev.get("payload") or {}
            if ev.get("type") == "turn_context" and p.get("model"):
                models.add(p["model"])
            t = p.get("type")
            if t in ("custom_tool_call", "function_call"):
                calls += 1
                name = p.get("name")
                if name == "exec":
                    code = p.get("input") or ""
                    bad = audit_exec_js(code, agent, obs_re, act_re)
                    views += "tools.view_image" in code and bad is None
                    if bad:
                        flags.append({"tool": "exec", "arg": f"{bad} :: {code[:300]}"})
                else:
                    flags.append({"tool": name, "arg": str(p.get("arguments") or p.get("input"))[:200]})
    for tr in sorted(glob.glob(os.path.join(priv, "transcript_*.jsonl"))):
        launches += 1
        for ev in load_events(tr):
            if ev.get("type") == "turn.completed":
                for k in usage:
                    usage[k] += int((ev.get("usage") or {}).get(k) or 0)
            elif ev.get("type") in ("turn.failed", "error"):
                errors += 1
    return {"tool_calls": calls, "image_views": views, "flags": flags, "permission_denials": 0,
            "launches": launches, "api_errors": errors, "claude_turns": 0, "cost_usd": 0.0,
            "duration_min": 0.0, "models": ",".join(sorted(models)), **usage}


def summarize(run):
    rows, audit = [], {}
    for priv in sorted(glob.glob(os.path.join(run, "private", "L*_*"))):
        name = os.path.basename(priv)
        meta = json.load(open(os.path.join(priv, "meta.json")))
        agent = os.path.join(run, "agent", name)
        is_codex = bool(glob.glob(os.path.join(priv, "rollout_*.jsonl")))
        au = audit_codex(priv, agent) if is_codex else audit_episode(priv, agent)
        audit[name] = au
        rows.append({"episode": name, "level": meta["level"], "difficulty": meta["difficulty"],
                     "n_pieces": meta["n_pieces"], "status": meta["status"],
                     "success": int(meta["status"] == "win"), "actions_used": meta["n_actions"],
                     "failed_actions": meta["n_failed"],
                     **{k: v for k, v in au.items() if k != "flags"},
                     "flagged_calls": len(au["flags"])})
    if not rows:
        print(f"no episodes under {run}/private")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(run, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    json.dump(audit, open(os.path.join(run, "audit.json"), "w"), indent=1)

    print(f"== {run}")
    for diff in ("easy", "mid", "hard", "all"):
        rs = [r for r in rows if diff == "all" or r["difficulty"] == diff]
        if not rs:
            continue
        acts = sum(r["actions_used"] for r in rs)
        print(f"{diff:5s} n={len(rs):3d}  success={sum(r['success'] for r in rs):3d} "
              f"({100 * sum(r['success'] for r in rs) / len(rs):5.1f}%)  "
              f"stopped_early={sum(r['status'] == 'playing' for r in rs):2d}  "
              f"avg_actions={acts / len(rs):5.2f}  "
              f"failed_action_rate={100 * sum(r['failed_actions'] for r in rs) / max(acts, 1):5.1f}%  "
              f"cost=${sum(r['cost_usd'] for r in rs):.2f}")
    print(f"models: {sorted({m for r in rows for m in r['models'].split(',') if m})}  "
          f"api_errors={sum(r['api_errors'] for r in rows)}  "
          f"flagged_calls={sum(r['flagged_calls'] for r in rows)}  "
          f"permission_denials={sum(r['permission_denials'] for r in rows)}")
    for name, au in audit.items():
        for fl in au["flags"]:
            print(f"FLAG {name}: {fl['tool']} {fl['arg']}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        summarize(os.path.abspath(r))
