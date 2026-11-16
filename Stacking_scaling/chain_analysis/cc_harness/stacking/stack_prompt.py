"""Closed-loop Stacking prompt for Claude Code.

The task text and tool descriptions are taken from the chainbench stacking task
(StackingGameTask._get_initial_instruction and StackingGameEnvironment.get_tool_schemas),
so the agent gets the same rules as the Kimi-k2.5 runs; only the interface part
(how to act, how to see observations) is new.

Usage:  python stack_prompt.py <session_dir> [--variant base|plan]
"""
import argparse
import json
import os

HEAD = """You are being evaluated as a player of a 3D packing puzzle, in the CLOSED-LOOP setting: you act one move at a time and see the result after every move.

## How to interact
- Your working folder holds the rendered observations of the game. obs_000.png shows the start state: the box on the left and every piece (in its initial pose) on the right. Look at it with {viewer}.
- Act with exactly this shell command, ONE action per call:
    stack-act '<one JSON action>'
  for example: stack-act '{{"tool":"place","piece_id":"3","cells":[[1,1,2],[2,1,2]]}}'
- Never chain several stack-act calls in one command (with ;, &&, | or newlines): only the first one is played, the rest are refused.
- After each action you get the tool result, the name of the new observation image (obs_<n>.png) and the current piece list. Look at the new image before choosing your next action.
- Budget: at most {max_actions} actions in total; every stack-act call counts, including failed ones. The game is won automatically as soon as the box is completely filled.
- Rules of fair play: the ONLY files you may read are the obs_*.png images in your working folder, and the ONLY command you may run is stack-act. Do not list, open or search any other file or folder, do not run Python or any other program, and do not write files.
{extra}{variant}
## Tools (the "tool" field of the JSON action)
- state: Show current game state: box size, occupancy, and placed/unplaced pieces. Example: {{"tool":"state"}}
- place: Place a piece using explicit cells. Provide all occupied cells in 1-based coordinates; length must match the piece voxel count. Example: {{"tool":"place","piece_id":"0","cells":[[1,1,1],[1,2,1],[2,1,1],[2,1,2]]}} (verify these cells form the same shape as the chosen piece; different valid cell layouts can represent rotations).
- pickup: Pick up a placed piece so it can be repositioned. Use this to remove a previously placed block when it was placed incorrectly or needs adjusting. Example: {{"tool":"pickup","piece_id":"0"}}

## The puzzle
**Goal:** Pack every piece into the **{size}** box for puzzle **`{puzzle_id}`**. You must fill **all {cells} cells** with **no collisions** and **no out-of-bounds** placements.

**Critical rules:**

1. Use the current image / board progress as the ground-truth state of what is already placed and what remains.
2. Think in 3D and plan explicitly: build a mental model of the box as (x, y, z) layers, check cross-sections layer-by-layer, and reason about connectivity/voids (avoid creating unreachable 1-cell cavities). Prefer a strategy like anchor corners/edges -> fill constrained regions -> resolve remaining gaps, and consider symmetry/rotations to match piece geometry.
3. Work in a safe order: inspect pieces -> plan a feasible sequence -> place pieces step-by-step, verifying after each placement (collision-free, in-bounds, and consistent with your 3D plan).
4. If a placement fails (collision / invalid) or you realize it blocks the remaining space, backtrack: use pickup to remove the previously placed piece (or the piece causing the conflict), then try a different orientation/order and re-place.

## Start state
{start}

## When you finish
Stop when the game is won, when the action budget is used up, or when you decide you cannot win. Then reply with one short paragraph: whether you won, how many actions it took, and in one sentence what made the puzzle hard.
"""

VARIANTS = {
    "base": "",
    "plan": (
        "- Planning: before EVERY stack-act call, write out your plan for the next several "
        "actions (at least three, unless the game ends sooner) and what you expect each to do; "
        "then execute only the first action of that plan. Revise the plan when a result "
        "or image contradicts it.\n"
    ),
}


VIEWERS = {"claude": "the Read tool", "codex": "your view_image tool"}
# Codex models run every tool through a JavaScript `exec` tool (code mode cannot be turned
# off for them), so the rule against computing the answer with code is stated explicitly.
EXTRA = {
    "claude": "",
    "codex": ("- Each tool call may contain exactly ONE tool use: either one view_image of an obs_*.png "
              "image or one stack-act command. Do not write any other code (no loops, arithmetic, "
              "search or helper functions) and do not start sub-agents: all reasoning about the "
              "puzzle must be your own.\n"),
}


def build(session: str, variant: str = "base", agent: str = "claude") -> str:
    meta = json.load(open(os.path.join(session, "meta.json")))
    start = open(os.path.join(session, "start.txt")).read().strip()
    a, b, c = (int(v) for v in meta["size"].split("x"))
    return HEAD.format(max_actions=meta["max_actions"], variant=VARIANTS[variant],
                       size=meta["size"], puzzle_id=meta["puzzle_id"], cells=a * b * c,
                       start=start, viewer=VIEWERS[agent], extra=EXTRA[agent])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ap.add_argument("--agent", default="claude", choices=sorted(VIEWERS))
    o = ap.parse_args()
    print(build(o.session, o.variant, o.agent), end="")
