"""
One CHAIN level (FluidNet, Domino, PipeFit) as a playable session for an
outside agent (a model, a person).

The game state lives in the session folder (`.state.json`); the agent only
ever sees rendered images and a short result, never the level file.  Every
call returns a `Result` (structured fields + the text a CLI prints), so an
agent harness can use the API directly or through `play_cli`.

    s, r = Session.start(level_dir, session_dir)          # renders the start state
    r = s.act(["open", [2, 3, 1]])                         # closed loop: one action
    r = s.view("top")                                      # look from another camera
    r = Session.open(session_dir).plan([[...], [...]])     # one-shot: whole plan
    specs = tool_specs(s.game)                             # tool definitions for an LLM

Observations: after every action the state is rendered from the session's
observation cameras (default front + back, `obs_<n>_<camera>.png`).  `view`
renders the current state from any camera in CAMERAS without changing the
game; views are logged and cost `view_cost` actions each (default 0).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .envs import game_of

STATE = ".state.json"
CAMERAS = ("front", "back", "top")
DEFAULT_VIEWS = ("front", "back")
MODS = {"fluidnet": "type_3", "domino": "type_3_1", "pipefit": "type_2_3"}

ACTIONS = {
    "fluidnet": '["open", [x, y, z]] opens the valve at that cell (opening is permanent)',
    "domino": ('["place", [x, y, z]] puts a patch domino with its base at that cell; '
               '["remove", [x, y, z]] takes it back; ["submit_placement"] ends the '
               'placing phase; ["push", "+x" | "-x" | "+y" | "-y"] pushes START once '
               'and decides the game'),
    "pipefit": ('["place", "I" | "L" | "T", [x, y, z], ["+x", "-z", ...]] places a pipe '
                'with those open ends; ["place", "S1" | "S2", [x, y, z]] places a support; '
                '["remove", [x, y, z]] takes a piece back; ["submit"] checks the network'),
}


@dataclass
class Result:
    """Outcome of one call.  `kind`: start, ok, rejected, invalid, refused,
    reset, quit, plan, view.  `status`: "" (playing), win, lose, quit."""
    kind: str
    text: str
    status: str = ""
    code: str = ""
    images: List[str] = field(default_factory=list)
    actions_used: int = 0
    max_actions: int = 0
    fails: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Game mechanics: replay an episode through each game package
# ---------------------------------------------------------------------------

def replay(game: str, level_dir: str, actions: List[list]):
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


def normalise(game: str, a: list) -> list:
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


def terminal(gs, last) -> str:
    """'win' | 'lose' | '' for the live episode."""
    if last is None:
        return ""
    if last.code.value.upper() == "WIN":
        return "win"
    return "lose" if gs.terminal else ""


def render(game: str, gs, path: str, title: str, camera: str) -> str:
    vis = __import__(f"{MODS[game]}.visualizer_plotly", fromlist=["render_state"])
    for attempt in range(5):             # kaleido fails now and then under load
        try:
            vis.render_state(gs, save_path=path, title=title, view=camera)
            return path
        except ValueError:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)
    return path


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, path: str, st: Dict):
        self.path = path
        self.st = st

    # -- creation / persistence ---------------------------------------------
    @classmethod
    def start(cls, level_dir: str, path: str, max_actions: int = 40,
              views: Sequence[str] = DEFAULT_VIEWS, view_cost: int = 0
              ) -> Tuple["Session", Result]:
        for c in views:
            if c not in CAMERAS:
                raise ValueError(f"unknown camera {c!r}; cameras: {', '.join(CAMERAS)}")
        os.makedirs(path, exist_ok=True)
        st = {"level": os.path.abspath(level_dir), "game": game_of(level_dir),
              "actions": [], "n_actions": 0, "fails": 0, "status": "",
              "max_actions": max_actions, "log": []}
        if tuple(views) != DEFAULT_VIEWS:
            st["obs_views"] = list(views)
        if view_cost:
            st["view_cost"] = view_cost
        s = cls(path, st)
        gs, _ = replay(s.game, st["level"], [])
        s._save()
        return s, s._observe(gs, "initial", f"started {s.game}", "start")

    @classmethod
    def open(cls, path: str) -> "Session":
        return cls(path, json.load(open(os.path.join(path, STATE))))

    def _save(self) -> None:
        json.dump(self.st, open(os.path.join(self.path, STATE), "w"), indent=1)

    @property
    def game(self) -> str:
        return self.st["game"]

    @property
    def obs_views(self) -> List[str]:
        return self.st.get("obs_views", list(DEFAULT_VIEWS))

    def episode(self) -> List[list]:
        """Actions since the last reset (the live episode)."""
        out: List[list] = []
        for a in self.st["actions"]:
            out = [] if a[0] == "reset" else out + [a]
        return out

    # -- results ---------------------------------------------------------------
    def _result(self, kind: str, text: str, code: str = "",
                images: Optional[List[str]] = None) -> Result:
        st = self.st
        return Result(kind=kind, text=text, status=st["status"], code=code,
                      images=images or [], actions_used=st["n_actions"],
                      max_actions=st["max_actions"], fails=st["fails"])

    def _report(self, msg: str, paths: List[str]) -> str:
        st = self.st
        lines = [msg,
                 f"status: {st['status'] or 'playing'}   actions used {st['n_actions']}/"
                 f"{st['max_actions']}   failed attempts {st['fails']}",
                 "observation: " + "  ".join(paths)]
        if st["status"] == "lose" and st["max_actions"] - st["n_actions"] > 0:
            lines.append('the game is over; send ["reset"] to try again, or ["quit"]')
        return "\n".join(lines)

    def _observe(self, gs, title: str, msg: str, kind: str, code: str = "") -> Result:
        n = len(self.st["actions"])
        paths = [render(self.game, gs, os.path.join(self.path, f"obs_{n:03d}_{c}.png"), title, c)
                 for c in self.obs_views]
        return self._result(kind, self._report(msg, paths), code, paths)

    # -- closed loop -----------------------------------------------------------
    def act(self, action) -> Result:
        """One action; renders the new state.  `action` is a list or JSON text."""
        st, game = self.st, self.game
        if st.get("oneshot"):
            return self._result("refused", "one-shot session: the plan has been played; no more actions")
        if st["status"] in ("win", "quit") or st["n_actions"] >= st["max_actions"]:
            return self._result("refused", f"episode finished ({st['status'] or 'out of actions'}); "
                                            "no more actions")
        try:
            a = normalise(game, json.loads(action) if isinstance(action, str) else action)
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            return self._result("invalid", f"invalid action: {e}")
        if a[0] == "quit":
            st["status"] = "quit"
            st["log"].append({"action": a, "result": "quit"})
            self._save()
            return self._result("quit", "episode ended (quit)")
        if a[0] == "reset":
            st["actions"].append(a)
            st["status"] = ""
            gs, _ = replay(game, st["level"], [])
            st["log"].append({"action": a, "result": "reset"})
            self._save()
            return self._observe(gs, "initial (reset)", "level reset", "reset")
        if st["status"] == "lose":
            return self._result("refused", 'the game is over; send ["reset"] to try again, or ["quit"]')
        ep = self.episode()
        try:
            gs, last = replay(game, st["level"], ep + [a])
        except Exception as e:                       # malformed coordinates etc.
            return self._result("invalid", f"invalid action: {type(e).__name__}: {e}")
        st["n_actions"] += 1
        code = last.code.value if last is not None else "OK"
        if code.upper() not in ("OK", "WIN") and not gs.terminal:   # rejected: state unchanged
            st["log"].append({"action": a, "result": code, "rejected": True})
            self._save()
            return self._result("rejected", f"rejected: {code} — {last.message}\n"
                                             f"actions used {st['n_actions']}/{st['max_actions']}", code)
        st["actions"].append(a)
        st["status"] = terminal(gs, last)
        if st["status"] == "lose":
            st["fails"] += 1
        detail = getattr(last, "detail", None) or {}
        msg = f"{code}: {last.message}" + (f"  {detail}" if detail and st["status"] else "")
        st["log"].append({"action": a, "result": code})
        self._save()
        return self._observe(gs, f"step {len(ep) + 1}: {json.dumps(a)} -> {code}", msg, "ok", code)

    # -- one-shot ----------------------------------------------------------------
    def plan(self, plan) -> Result:
        """Play a whole action list from the start state, then close the
        session.  Rejected actions are skipped (they still count); `reset`
        and `quit` are not allowed.  `plan` is a list or JSON text."""
        st, game = self.st, self.game
        if st.get("oneshot") or st["actions"] or st["log"]:     # views may come first
            return self._result("refused", "this session already has actions; "
                                           "a plan must be the first and only call")
        try:
            plan = json.loads(plan) if isinstance(plan, str) else plan
            if not isinstance(plan, list) or not all(isinstance(a, list) for a in plan):
                raise ValueError("a plan is a JSON list of actions")
        except (ValueError, json.JSONDecodeError) as e:
            return self._result("invalid", f"invalid plan: {e}")   # nothing used: may be repeated
        st["oneshot"] = True
        gs, last = replay(game, st["level"], [])
        for raw_a in plan:
            if st["n_actions"] >= st["max_actions"] or st["status"]:
                break
            st["n_actions"] += 1
            try:
                a = normalise(game, raw_a)
                if a[0] in ("reset", "quit"):
                    raise ValueError(f"{a[0]} is not allowed in a plan")
                gs, last = replay(game, st["level"], st["actions"] + [a])
            except Exception as e:
                st["log"].append({"action": raw_a, "result": f"invalid: {e}", "rejected": True})
                continue
            code = last.code.value if last is not None else "OK"
            if code.upper() not in ("OK", "WIN") and not gs.terminal:
                st["log"].append({"action": a, "result": code, "rejected": True})
                continue
            st["actions"].append(a)
            st["status"] = terminal(gs, last)
            st["log"].append({"action": a, "result": code})
        if st["status"] == "lose":
            st["fails"] += 1
        st["plan"] = plan
        self._save()
        gs, _ = replay(game, st["level"], st["actions"])
        finals = []
        try:                                        # final state, for inspection only
            for c in self.obs_views:
                finals.append(render(game, gs, os.path.join(self.path, f"final_{c}.png"), "final", c))
        except ValueError:
            pass
        return self._result("plan", f"plan received ({len(plan)} actions); the episode is closed",
                            images=finals)

    # -- cameras -----------------------------------------------------------------
    def view(self, camera: str) -> Result:
        """Render the live state from `camera` without changing the game.
        Allowed before a plan and during a closed-loop episode; costs
        `view_cost` actions (default 0) and is logged in `views`."""
        st = self.st
        if camera not in CAMERAS:
            return self._result("invalid", f"unknown camera {camera!r}; cameras: {', '.join(CAMERAS)}")
        if st.get("oneshot"):
            return self._result("refused", "one-shot session: the plan has been played; no more views")
        cost = st.get("view_cost", 0)
        if cost and st["n_actions"] + cost > st["max_actions"]:
            return self._result("refused", "not enough actions left for a view")
        st["n_actions"] += cost
        gs, _ = replay(self.game, st["level"], self.episode())
        n = len(st["actions"])
        path = render(self.game, gs, os.path.join(self.path, f"obs_{n:03d}_{camera}.png"),
                      f"view: {camera}", camera)
        st.setdefault("views", []).append({"camera": camera, "after_actions": n})
        self._save()
        return self._result("view", f"view {camera}\nobservation: {path}", images=[path])

    # -- bookkeeping ------------------------------------------------------------
    def status(self) -> Dict:
        """What the player may know: no level contents."""
        st = self.st
        return {"game": self.game, "status": st["status"] or "playing",
                "actions_used": st["n_actions"], "max_actions": st["max_actions"],
                "failed_attempts": st["fails"], "oneshot": bool(st.get("oneshot")),
                "views": len(st.get("views", [])), "cameras": list(CAMERAS),
                "observation_cameras": self.obs_views}

    def export_rows(self, model: str) -> Tuple[Dict, Dict]:
        """(exp1 trajectory line, exp2 result line)."""
        st = self.st
        level = os.path.basename(st["level"])
        traj = {"model": model, "level": level, "actions": st["actions"]}
        res = {"model": model, "level": level, "win": st["status"] == "win",
               "verifies": st["fails"] + (st["status"] == "win"),
               "actions": st["n_actions"],
               "rejected": sum(1 for l in st["log"] if l.get("rejected")),
               "views": len(st.get("views", []))}
        return traj, res


# ---------------------------------------------------------------------------
# Tool definitions for an LLM agent (name, description, JSON-schema input)
# ---------------------------------------------------------------------------

def tool_specs(game: str, oneshot: bool = False) -> List[Dict]:
    """Tools an agent gets for one level: `act` (closed loop) or `plan`
    (one-shot), plus `view`.  Input schemas are JSON Schema."""
    action_help = ACTIONS[game]
    view = {"name": "view",
            "description": ("Render the current state from another camera without "
                            "changing the game: " + ", ".join(CAMERAS) + ". Returns an image."),
            "input_schema": {"type": "object",
                             "properties": {"camera": {"type": "string", "enum": list(CAMERAS)}},
                             "required": ["camera"]}}
    if oneshot:
        main = {"name": "plan",
                "description": ("Submit the whole solution at once as a list of actions; it is "
                                "played from the start with no feedback and the episode closes. "
                                "Actions: " + action_help + "."),
                "input_schema": {"type": "object",
                                 "properties": {"actions": {"type": "array",
                                                            "items": {"type": "array"}}},
                                 "required": ["actions"]}}
    else:
        main = {"name": "act",
                "description": ("Play one action and get the new state's images. Actions: "
                                + action_help + '; ["reset"] restarts after a loss, '
                                '["quit"] gives up.'),
                "input_schema": {"type": "object",
                                 "properties": {"action": {"type": "array"}},
                                 "required": ["action"]}}
    return [main, view]
