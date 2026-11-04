"""
Command line for playing one CHAIN level as an outside agent (a model, a
person); a thin wrapper around `chain_analysis.session.Session`.

The game state lives in the session folder; the agent only ever sees the
rendered observation and a one-line result, never the level file.

    python -m chain_analysis.play_cli start <level_dir> <session_dir> [--max-actions 40]
                                      [--views front,back] [--view-cost 0]
    python -m chain_analysis.play_cli act <session_dir> '<json action>'
    python -m chain_analysis.play_cli view <session_dir> front|back|top
    python -m chain_analysis.play_cli plan <session_dir> '<json list of actions>'
    python -m chain_analysis.play_cli status <session_dir>
    python -m chain_analysis.play_cli tools fluidnet|domino|pipefit [--oneshot]
    python -m chain_analysis.play_cli export <session_dir>... --model NAME --out runs.jsonl

Actions (JSON lists):
  FluidNet  ["open", [x, y, z]]
  Domino    ["place", [x, y, z]]   ["remove", [x, y, z]]   ["submit_placement"]
            ["push", "+x" | "-x" | "+y" | "-y"]     (push, then the verdict)
  PipeFit   ["place", "I" | "L" | "T", [x, y, z], ["+x", "-z", ...]]  (ends)
            ["place", "S1" | "S2", [x, y, z]]   ["remove", [x, y, z]]   ["submit"]
  all       ["reset"]  start the level over after a loss
            ["quit"]   give up

Every action renders `obs_<n>_<camera>.png` for the session's observation
cameras (default front + back) and prints the result.  `view` renders the
current state from another camera (front, back, top) without changing the
game; it costs `--view-cost` actions (default 0) and is logged.  `plan` is
the one-shot protocol: the whole action list is played at once from the
start state without intermediate observations, and the session is then
closed (views are allowed before it).  The game rules are the real
mechanics of each package.  `export` writes the trajectories in exp1's
format and a per-episode result line for exp2 (`--results`).
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .session import (CAMERAS, DEFAULT_VIEWS, MODS, Session, normalise, replay,
                      tool_specs)

VIEWS = {g: DEFAULT_VIEWS for g in MODS}       # default observation cameras
_replay, _normalise = replay, normalise          # older scripts import these


def cmd_start(level_dir: str, session: str, max_actions: int,
              views=DEFAULT_VIEWS, view_cost: int = 0) -> None:
    try:
        print(Session.start(level_dir, session, max_actions, views, view_cost)[1].text)
    except ValueError as e:
        print(f"cannot start: {e}")


def cmd_act(session: str, raw: str) -> None:
    print(Session.open(session).act(raw).text)


def cmd_view(session: str, camera: str) -> None:
    print(Session.open(session).view(camera).text)


def cmd_plan(session: str, raw: str) -> None:
    print(Session.open(session).plan(raw).text)


def cmd_export(sessions: List[str], model: str, out: str, results: str) -> None:
    with open(out, "a") as f, open(results, "a") as g:
        for s in sessions:
            traj, res = Session.open(s).export_rows(model)
            f.write(json.dumps(traj) + "\n")
            g.write(json.dumps(res) + "\n")
    print(f"{len(sessions)} episodes -> {out}, {results}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("level"); p.add_argument("session")
    p.add_argument("--max-actions", type=int, default=40)
    p.add_argument("--views", default=",".join(DEFAULT_VIEWS),
                   help=f"observation cameras after each action, from {', '.join(CAMERAS)}")
    p.add_argument("--view-cost", type=int, default=0,
                   help="actions charged per `view` call")
    p = sub.add_parser("act")
    p.add_argument("session"); p.add_argument("action")
    p = sub.add_parser("view")
    p.add_argument("session"); p.add_argument("camera", choices=CAMERAS)
    p = sub.add_parser("plan")
    p.add_argument("session"); p.add_argument("plan")
    p = sub.add_parser("status")
    p.add_argument("session")
    p = sub.add_parser("tools")
    p.add_argument("game", choices=sorted(MODS)); p.add_argument("--oneshot", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "start":
        cmd_start(a.level, a.session, a.max_actions, a.views.split(","), a.view_cost)
    elif a.cmd == "act":
        cmd_act(a.session, a.action)
    elif a.cmd == "view":
        cmd_view(a.session, a.camera)
    elif a.cmd == "plan":
        cmd_plan(a.session, a.plan)
    elif a.cmd == "status":
        print(json.dumps(Session.open(a.session).status(), indent=1))
    elif a.cmd == "tools":
        print(json.dumps(tool_specs(a.game, a.oneshot), indent=1, ensure_ascii=False))
    else:
        cmd_export(a.sessions, a.model, a.out, a.results)


if __name__ == "__main__":
    main()
