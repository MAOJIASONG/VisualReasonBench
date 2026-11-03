"""One-shot prompts for a model that gets the start images attached and answers with the plan."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sonnet_oneshot50"))
from make_prompts import GAMES

HEAD = """You are being evaluated as a player of a 3D puzzle game called {name}, in the ONE-SHOT setting: you see only the start state and must give your complete solution at once. There is no feedback while it is played.

The attached image{s} {are} the start state{views}. Study {them} as long as you need; read the coordinates off the axis labels carefully.

Your answer is played from the start, action by action, through the real game rules. An action the game rejects (occupied cell, wrong coordinates, ...) is skipped and the rest of the plan continues. The plan may hold at most 40 actions.

Rules: answer from the image{s} alone. Do not run any shell command, do not read or list any file, and do not use any tool.

## The game
"""

TAIL = """
## Your answer
Think it through, then end your reply with the plan as a single JSON list of actions on the LAST line, for example:
{example}
"""

def main(O):
    os.makedirs(os.path.join(O, "prompts"), exist_ok=True)
    for line in open(os.path.join(O, "map.txt")):
        sess, _ = line.split()
        num, game = os.path.basename(sess).split("_", 1)
        name, rules, images, views, ex = GAMES[game]
        two = "back" in images
        txt = HEAD.format(name=name, s="s" if two else "", are="are" if two else "is",
                          views=" (front and back camera views of the same state)" if two else " (front camera view)",
                          them="them" if two else "it") + rules + TAIL.format(example=ex)
        open(os.path.join(O, "prompts", num + ".md"), "w").write(txt)

if __name__ == "__main__":
    main(sys.argv[1])
