"""Closed-loop Puzzle (Luban lock) prompt for Claude Code / Codex.

The task text is the benchmark's (LubanDisassemblyTask system prompt + instruction) and the
tools are those of LubanEnvironment.get_tool_schemas (move_piece, rotate_piece); the interface
part (how to act and see observations) matches the Stacking and Experiment 6 harnesses. The
game's win rule (pieces must end up in the removal zone) is stated, since the game only counts
a piece as removed there.

Usage:  python luban_prompt.py <session_dir> [--variant base|plan] [--agent claude|codex]
"""
import argparse
import json
import os

HEAD = """You are a spatial reasoning expert solving a "Luban Lock" (a 3D interlocking burr puzzle), in the CLOSED-LOOP setting: you act one move at a time and see the result after every move.

## How to interact
- Your working folder holds the rendered observations. obs_000.png shows the start state from three cameras (VIEW 1, VIEW 3, VIEW 5). Look at it with {viewer}.
- Act with exactly this shell command, ONE action per call:
    luban-act '<one JSON action>'
  for example: luban-act '{{"tool":"move_piece","piece_id":2,"axis":"x","distance":30}}'
- Never chain several luban-act calls in one command (with ;, &&, | or newlines): only the first one is played, the rest are refused.
- After each action you get the result (success, or failed = blocked and nothing changed), the name of the new observation image (obs_<n>.png) and the piece list (id, position, orientation, removed). Look at the new image before choosing your next action.
- Budget: at most {max_actions} actions in total; every luban-act call counts, including failed ones.
- Rules of fair play: the ONLY files you may read are the obs_*.png images in your working folder, and the ONLY command you may run is luban-act. Do not list, open or search any other file or folder, do not run Python or any other program, and do not write files.
{extra}{variant}
## Tools (the "tool" field of the JSON action)
- move_piece: Move a piece along an axis by a distance in logic units. Fields: piece_id (integer), axis ("x", "y" or "z"), distance (number; negative = towards -axis), optional is_exploratory (true = move in that direction until blocked, up to the distance). A move that would make the piece collide is refused and nothing moves.
- rotate_piece: Rotate a piece by multiples of 90 degrees around an axis through the piece. Fields: piece_id, axis ("x", "y" or "z"), angle (degrees, a multiple of 90). Refused if the piece would collide while turning.

Coordinates: Y is vertical (up), X and Z are horizontal; positions and distances are in the game's logic units (the pieces of this puzzle are roughly {size_hint} units across).

## The puzzle
Pieces are mechanically interlocked and must be moved via the tools. The puzzle is LOCKED: identify the KEY piece (the unblocked one) and slide it out along its free axis, then keep taking the puzzle apart.

**Goal:** remove pieces until only one is left in the assembly ({n_pieces} pieces in total, so {n_goal} must be removed). A piece counts as removed when it ends a move inside the removal zone around the puzzle: between 150 and 450 units from the puzzle's centre along x, y or z (within 300 units on the other two axes). A piece moved much further than that skips the zone and does not count. Removed pieces stay where they are and still block other pieces. The game is won automatically as soon as {n_goal} pieces are removed.

## Start state
{start}

## When you finish
Stop when the game is won, when the action budget is used up, or when you decide you cannot win. Then reply with one short paragraph: whether you won, how many actions it took, and in one sentence what made the puzzle hard.
"""

VARIANTS = {
    "base": "",
    "plan": (
        "- Planning: before EVERY luban-act call, write out your plan for the next several "
        "actions (at least three, unless the game ends sooner) and what you expect each to do; "
        "then execute only the first action of that plan. Revise the plan when a result "
        "or image contradicts it.\n"
    ),
}

VIEWERS = {"claude": "the Read tool", "codex": "your view_image tool"}
EXTRA = {
    "claude": "",
    "codex": ("- Each tool call may contain exactly ONE tool use: either one view_image of an obs_*.png "
              "image or one luban-act command. Do not write any other code (no loops, arithmetic, "
              "search or helper functions) and do not start sub-agents: all reasoning about the "
              "puzzle must be your own.\n"),
}


def build(session, variant="base", agent="claude"):
    meta = json.load(open(os.path.join(session, "meta.json")))
    start = open(os.path.join(session, "start.txt")).read().strip()
    return HEAD.format(max_actions=meta["max_actions"], variant=VARIANTS[variant], viewer=VIEWERS[agent],
                       extra=EXTRA[agent], n_pieces=meta["n_pieces"], n_goal=meta["n_pieces"] - 1,
                       size_hint=meta.get("size_hint", "40-200"), start=start)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ap.add_argument("--agent", default="claude", choices=sorted(VIEWERS))
    a = ap.parse_args()
    print(build(a.session, a.variant, a.agent), end="")
