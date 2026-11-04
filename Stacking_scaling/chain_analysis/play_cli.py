"""
Closed-loop play of one CHAIN level by an outside agent (a model, a person).

The game state lives in the session folder; the agent only ever sees the
rendered observation and a one-line result, never the level file.

    python -m chain_analysis.play_cli start <level_dir> <session_dir> [--max-actions 40]
    python -m chain_analysis.play_cli act <session_dir> '<json action>'
    python -m chain_analysis.play_cli plan <session_dir> '<json list of actions>'
    python -m chain_analysis.play_cli export <session_dir>... --model NAME --out runs.jsonl

Actions (JSON lists):
  FluidNet  ["open", [x, y, z]]
  Domino    ["place", [x, y, z]]   ["remove", [x, y, z]]   ["submit_placement"]
            ["push", "+x" | "-x" | "+y" | "-y"]     (push, then the verdict)
  PipeFit   ["place", "I" | "L" | "T", [x, y, z], ["+x", "-z", ...]]  (ends)
            ["place", "S1" | "S2", [x, y, z]]   ["remove", [x, y, z]]   ["submit"]
  all       ["reset"]  start the level over after a loss
            ["quit"]   give up

Every call renders `obs_<n>_front.png` (and `_back.png` for FluidNet /
PipeFit) in the session folder and prints the result.  `plan` is the
one-shot protocol: the whole action list is played at once from the start
state without intermediate observations, and the session is then closed.  The game rules are
the real mechanics of each package.  `export` writes the trajectories in
exp1's format and a per-episode result line for exp2 (`--results`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

from .envs import game_of

STATE = ".state.json"
VIEWS = {"fluidnet": ("front", "back"), "pipefit": ("front", "back"),
         "domino": ("front", "back")}
MODS = {"fluidnet": "type_3", "domino": "type_3_1", "pipefit": "type_2_3"}


def _load(session: str) -> Dict:
    return json.load(open(os.path.join(session, STATE)))


def _save(session: str, st: Dict) -> None:
    json.dump(st, open(os.path.join(session, STATE), "w"), indent=1)


def _episode(st: Dict) -> List[list]:
    """Actions since the last reset (the live episode)."""
    out: List[list] = []
    for a in st["actions"]:
        out = [] if a[0] == "reset" else out + [a]
    return out


# -- per game: replay an episode through the real mechanics ----------------

def _replay(game: str, level_dir: str, actions: List[list]):
    """(GameState, result of the last action) after `actions`."""
    lvl = os.path.join(level_dir, "level.json")
    last = None
    if game == "fluidnet":
        from type_3.loader import create_game_state, load_level_from_json
        from type_3.propagation import step_open_valve
        gs = create_game_state(load_level_from_json(lvl))
        for a in actions:
            last = step_open_valve(gs, tuple(a[1]))
        return gs, last
    if game == "domino":
        from type_3_1.difficulty import start_of
        from type_3_1.game_core import PieceType, init_state
        from type_3_1.loader import load_level_from_json
        from type_3_1.mechanics import step_place, step_push, step_remove, submit, submit_placement
        spec = load_level_from_json(lvl)
        gs = init_state(spec)
        for a in actions:
            if a[0] == "place":
                last = step_place(gs, PieceType.DOMINO, tuple(a[1]))
            elif a[0] == "remove":
                last = step_remove(gs, tuple(a[1]))
            elif a[0] == "submit_placement":
                last = submit_placement(gs)
            elif a[0] == "push":
                last = step_push(gs, start_of(spec), a[1])
                if last.code.value == "OK":
                    last = submit(gs)
        return gs, last
    from type_2_3.game_core import PieceType, init_state
    from type_2_3.loader import load_level_from_json
    from type_2_3.mechanics import step_place, step_remove, submit
    gs = init_state(load_level_from_json(lvl))
    for a in actions:
        if a[0] == "place":
            last = step_place(gs, PieceType(a[1]), tuple(a[2]), a[3] if len(a) > 3 else 0)
        elif a[0] == "remove":
            last = step_remove(gs, tuple(a[1]))
        elif a[0] == "submit":
            last = submit(gs)
    return gs, last


def _normalise(game: str, a: list) -> list:
    """Validate the shape; PipeFit ends -> orientation index."""
    if not isinstance(a, list) or not a:
        raise ValueError("an action is a JSON list, e.g. [\"reset\"]")
    if a[0] in ("reset", "quit"):
        return [a[0]]
    if game == "pipefit" and a[0] == "place":
        from type_2_3.game_core import PieceType
        from type_2_3.mechanics import orientation_from_ends
        pt = PieceType(a[1])
        if a[1] in ("S1", "S2"):
            return ["place", a[1], list(a[2]), 0]
        if len(a) < 4 or not isinstance(a[3], list):
            raise ValueError('pipes need their ends, e.g. ["place", "L", [2,3,1], ["+x", "-y"]]')
        o = orientation_from_ends(pt, a[3])
        if o is None:
            raise ValueError(f"a {a[1]} pipe cannot have ends {a[3]}")
        return ["place", a[1], list(a[2]), o]
    return a


def _terminal(gs, last) -> str:
    """'win' | 'lose' | '' for the live episode."""
    if last is None:
        return ""
    if last.code.value.upper() == "WIN":
        return "win"
    return "lose" if gs.terminal else ""


def _render(st: Dict, gs, session: str, title: str) -> List[str]:
    game = st["game"]
    vis = __import__(f"{MODS[game]}.visualizer_plotly", fromlist=["render_state"])
    n = len(st["actions"])
    paths = []
    for v in VIEWS[game]:
        p = os.path.join(session, f"obs_{n:03d}_{v}.png")
        for attempt in range(5):         # kaleido fails now and then under load
            try:
                vis.render_state(gs, save_path=p, title=title, view=v)
                break
            except ValueError:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        paths.append(p)
    return paths


def _report(st: Dict, msg: str, paths: List[str]) -> None:
    left = st["max_actions"] - st["n_actions"]
    print(msg)
    print(f"status: {st['status'] or 'playing'}   actions used {st['n_actions']}/"
          f"{st['max_actions']}   failed attempts {st['fails']}")
    print("observation: " + "  ".join(paths))
    if st["status"] == "lose" and left > 0:
        print('the game is over; send ["reset"] to try again, or ["quit"]')


def cmd_start(level_dir: str, session: str, max_actions: int) -> None:
    os.makedirs(session, exist_ok=True)
    st = {"level": os.path.abspath(level_dir), "game": game_of(level_dir),
          "actions": [], "n_actions": 0, "fails": 0, "status": "",
          "max_actions": max_actions, "log": []}
    gs, _ = _replay(st["game"], st["level"], [])
    _save(session, st)
    paths = _render(st, gs, session, "initial")
    _report(st, f"started {st['game']}", paths)


def cmd_act(session: str, raw: str) -> None:
    st = _load(session)
    game = st["game"]
    if st.get("oneshot"):
        print("one-shot session: the plan has been played; no more actions")
        return
    if st["status"] in ("win", "quit") or st["n_actions"] >= st["max_actions"]:
        print(f"episode finished ({st['status'] or 'out of actions'}); no more actions")
        return
    try:
        a = _normalise(game, json.loads(raw))
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        print(f"invalid action: {e}")
        return
    if a[0] == "quit":
        st["status"] = "quit"
        st["log"].append({"action": a, "result": "quit"})
        _save(session, st)
        print("episode ended (quit)")
        return
    if a[0] == "reset":
        st["actions"].append(a)
        st["status"] = ""
        gs, _ = _replay(game, st["level"], [])
        st["log"].append({"action": a, "result": "reset"})
        _save(session, st)
        paths = _render(st, gs, session, "initial (reset)")
        _report(st, "level reset", paths)
        return
    if st["status"] == "lose":
        print('the game is over; send ["reset"] to try again, or ["quit"]')
        return
    ep = _episode(st)
    try:
        gs, last = _replay(game, st["level"], ep + [a])
    except Exception as e:                       # malformed coordinates etc.
        print(f"invalid action: {type(e).__name__}: {e}")
        return
    st["n_actions"] += 1
    code = last.code.value if last is not None else "OK"
    if code.upper() not in ("OK", "WIN") and not gs.terminal:   # rejected: state unchanged
        st["log"].append({"action": a, "result": code, "rejected": True})
        _save(session, st)
        print(f"rejected: {code} — {last.message}")
        print(f"actions used {st['n_actions']}/{st['max_actions']}")
        return
    st["actions"].append(a)
    st["status"] = _terminal(gs, last)
    if st["status"] == "lose":
        st["fails"] += 1
    detail = getattr(last, "detail", None) or {}
    msg = f"{code}: {last.message}" + (f"  {detail}" if detail and st["status"] else "")
    st["log"].append({"action": a, "result": code})
    _save(session, st)
    title = f"step {len(ep) + 1}: {json.dumps(a)} -> {code}"
    paths = _render(st, gs, session, title)
    _report(st, msg, paths)


def cmd_plan(session: str, raw: str) -> None:
    """One-shot: play a whole action list from the start, then close."""
    st = _load(session)
    game = st["game"]
    if st.get("oneshot") or st["actions"] or st["n_actions"]:
        print("this session already has actions; a plan must be the first and only call")
        return
    try:
        plan = json.loads(raw)
        if not isinstance(plan, list) or not all(isinstance(a, list) for a in plan):
            raise ValueError("a plan is a JSON list of actions")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"invalid plan: {e}")             # nothing used: the call can be repeated
        return
    st["oneshot"] = True
    gs, last = _replay(game, st["level"], [])
    for raw_a in plan:
        if st["n_actions"] >= st["max_actions"] or st["status"]:
            break
        st["n_actions"] += 1
        try:
            a = _normalise(game, raw_a)
            if a[0] in ("reset", "quit"):
                raise ValueError(f"{a[0]} is not allowed in a plan")
            gs, last = _replay(game, st["level"], st["actions"] + [a])
        except Exception as e:
            st["log"].append({"action": raw_a, "result": f"invalid: {e}", "rejected": True})
            continue
        code = last.code.value if last is not None else "OK"
        if code.upper() not in ("OK", "WIN") and not gs.terminal:
            st["log"].append({"action": a, "result": code, "rejected": True})
            continue
        st["actions"].append(a)
        st["status"] = _terminal(gs, last)
        st["log"].append({"action": a, "result": code})
    if st["status"] == "lose":
        st["fails"] += 1
    st["plan"] = plan
    _save(session, st)
    gs, _ = _replay(game, st["level"], st["actions"])
    vis = __import__(f"{MODS[game]}.visualizer_plotly", fromlist=["render_state"])
    try:                                        # final state, for inspection only
        for v in VIEWS[game]:
            vis.render_state(gs, save_path=os.path.join(session, f"final_{v}.png"),
                             title="final", view=v)
    except ValueError:
        pass
    print(f"plan received ({len(plan)} actions); the episode is closed")


def cmd_export(sessions: List[str], model: str, out: str, results: str) -> None:
    with open(out, "a") as f, open(results, "a") as g:
        for s in sessions:
            st = _load(s)
            level = os.path.basename(st["level"])
            f.write(json.dumps({"model": model, "level": level,
                                "actions": st["actions"]}) + "\n")
            g.write(json.dumps({"model": model, "level": level,
                                "win": st["status"] == "win",
                                "verifies": st["fails"] + (st["status"] == "win"),
                                "actions": st["n_actions"],
                                "rejected": sum(1 for l in st["log"] if l.get("rejected"))}) + "\n")
    print(f"{len(sessions)} episodes -> {out}, {results}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("level"); p.add_argument("session")
    p.add_argument("--max-actions", type=int, default=40)
    p = sub.add_parser("act")
    p.add_argument("session"); p.add_argument("action")
    p = sub.add_parser("plan")
    p.add_argument("session"); p.add_argument("plan")
    p = sub.add_parser("export")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "start":
        cmd_start(a.level, a.session, a.max_actions)
    elif a.cmd == "act":
        cmd_act(a.session, a.action)
    elif a.cmd == "plan":
        cmd_plan(a.session, a.plan)
    else:
        cmd_export(a.sessions, a.model, a.out, a.results)


if __name__ == "__main__":
    main()
