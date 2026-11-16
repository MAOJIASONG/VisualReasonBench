"""Closed-loop prompts for running Claude Code (or another CLI agent) on the CHAIN games.

The game rules are reused verbatim from the one-shot Sonnet run
(model_runs/sonnet_oneshot50/make_prompts.py) so results stay comparable.

Usage:  python prompts.py --game type_3 [--variant base|plan]
"""
import argparse
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "model_runs", "sonnet_oneshot50", "make_prompts.py")
_spec = importlib.util.spec_from_file_location("oneshot_prompts", _SRC)
_mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mp)

# game dir -> (display name, rules text, example action)
GAMES = {
    "type_3": ("FLUIDNET", _mp.FLUID, '["open", [2, 3, 1]]'),
    "type_3_1": ("DOMINO CHAIN", _mp.DOMINO, '["place", [2, 5, 1]]'),
    "type_2_3": ("PIPEFIT", _mp.PIPE, '["place", "I", [2, 3, 1], ["+x", "-x"]]'),
}

HEAD = """You are being evaluated as a player of a 3D puzzle game called {name}, in the CLOSED-LOOP setting: you act one move at a time and see the result after every move.

## How to interact
- Your working folder holds the rendered observations of the game. obs_000_front.png and obs_000_back.png show the start state (two camera views of the same state). Look at them with {viewer} and read the coordinates off the axis labels carefully.
- Act with exactly this shell command, ONE action per call:
    chain-act '<one JSON action>'
  for example: chain-act '{example}'
- After each action you get a one-line result and the names of the new observation images (obs_<n>_front.png, obs_<n>_back.png). Look at the new images before choosing your next action.
- Budget: at most 40 actions in total (rejected actions count too). If you lose, you may send chain-act '["reset"]' to restart the level from the start state; every loss counts as a failed attempt.
- Rules of fair play: the ONLY files you may read are the obs_*.png images in your working folder, and the ONLY command you may run is chain-act. Do not list, open or search any other file or folder, do not run Python or any other program, and do not write files.
{extra}{variant}
## The game
"""

# How each CLI agent views images, and agent-specific rules. Codex runs these models in
# "code mode" (every tool call is a small JavaScript program), which cannot be switched off,
# so the prompt forbids using it for anything but one image view or one chain-act per call.
VIEWERS = {"claude": "the Read tool", "codex": "your view_image tool"}
EXTRA = {
    "claude": "",
    "codex": (
        "- Each tool call may contain exactly ONE tool use: either one view_image of an obs_*.png "
        "image or one chain-act command. Do not write any other code (no loops, arithmetic, search "
        "or helper functions) and do not start sub-agents: all reasoning about the puzzle must be "
        "your own.\n"
        "- Never chain several chain-act calls in one command (with ;, &&, | or newlines): only the "
        "first one is played, the rest are refused.\n"
    ),
}

VARIANTS = {
    "base": "",
    # Experiment 6, "explicit planning" harness.
    "plan": (
        "- Planning: before EVERY chain-act call, write out your plan for the next several "
        "actions (at least three, unless the game ends sooner) and what you expect each to do; "
        "then execute only the first action of that plan. Revise the plan when a result "
        "or image contradicts it.\n"
    ),
}

TAIL = """
## When you finish
Stop when the game is won, when the action budget is used up, or when you decide you cannot win. Then reply with one short paragraph: whether you won, how many attempts it took, and in one sentence what made the level hard.
"""


def build(game: str, variant: str = "base", agent: str = "claude") -> str:
    name, rules, example = GAMES[game]
    # The shared rules end with a "Your plan:" line written for the one-shot setting.
    rules = rules.replace("- Your plan:", "- A typical full action sequence:")
    return HEAD.format(name=name, example=example, variant=VARIANTS[variant],
                       viewer=VIEWERS[agent], extra=EXTRA[agent]) + rules + TAIL


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(GAMES))
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ap.add_argument("--agent", default="claude", choices=sorted(VIEWERS))
    a = ap.parse_args()
    print(build(a.game, a.variant, a.agent), end="")
