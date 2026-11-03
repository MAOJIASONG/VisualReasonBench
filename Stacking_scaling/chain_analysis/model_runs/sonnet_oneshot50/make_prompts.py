import os, sys
CLI = "python3.11 -m chain_analysis.play_cli plan"

HEAD = """You are being evaluated as a player of a 3D puzzle game called {name}, in the ONE-SHOT setting: you see only the start state and must give your complete solution in a single command. There is no feedback while it is played.

## How to interact
Working directory: /mnt/hdfs/yuhao/VisualReasonBench/Stacking_scaling
Your session folder: {sess}

- Look at the start state with the Read tool: {images} in your session folder{views}. Study it as long as you need; read the coordinates off the axis labels carefully.
- Then submit your whole plan ONCE, as a JSON list of actions, with exactly this Bash command (from the working directory):
  {cli} {sess} '<JSON list of actions>'
  for example: {cli} {sess} '{example}'
- The plan is played from the start, action by action, through the real game rules. You get no image and no result back; the session is closed after this one call. A plan that is not valid JSON is refused without cost, so you may send it again; otherwise you get exactly one call.
- An action the game rejects (occupied cell, wrong coordinates, ...) is skipped and the rest of the plan continues. The plan may hold at most 40 actions.
- Rules of fair play: the ONLY files you may read are this instruction file and the obs_000_*.png images in your session folder, and the ONLY command you may run is the play_cli `plan` command above, once. Do not open, list, grep or cat any other file or folder (no level files, no source code, no other sessions, not the hidden .state.json). Do not run Python or any other program, and do not write files.

## The game
"""

TAIL = """
## When you finish
After the plan command, stop. Reply with one short paragraph: the plan you sent, how confident you are that it wins, and in one sentence what made the level hard to plan from one image.
"""

FLUID = """- The board is a 3D grid of cells with 1-based coordinates (x, y, z); z points up. The axes are labelled on the floor edges and on the side (z=1, z=2, ...).
- Sources (big spheres on a pump) are where fluid starts; some levels have two pumps of different colours. Pipes (dark tubes) are always open. Valves (amber cubes with a hand-wheel) block flow while closed.
- Each action opens ONE valve (permanent): '["open", [x, y, z]]'. After each opening, fluid floods everything connected to a source through pipes and open valves.
- Win: every target tank (glass box) is filled with the colour of its frame. The game stops at the first win, so actions after it do not matter.
- Lose (game over at once, the rest of the plan is not played): fluid touches a red protected cell, fluid enters a tank of the wrong colour, or two colours meet. A pipe stub sticking out of a red cell shows where its feed comes from.
- A valve that looks harmless can connect to a red cell further along the pipes, and the ORDER of openings matters.
- Your plan: the list of valves to open, in order, e.g. [["open", [2, 3, 1]], ["open", [4, 3, 1]]].
"""

DOMINO = """- A 3D board with 1-based coordinates (x, y, z); z points up, axes are labelled on the floor edges and the side. Every domino is two cells tall; its position is its base cell.
- Pieces: green START (the only piece you push), cream dominoes D (fall forward and knock the next piece), cream splitters T (send the wave out to both sides), gold goals G (topple every one to win), red protected P (toppling one loses), grey steps (blocks that hold a piece one level up).
- A falling piece hits the piece in the next cell in its fall direction if that piece stands at most one level higher or lower.
- Phase 1 — place: you have a fixed number of patch dominoes (see the PATCH DOMINOES panel) and must place ALL of them before pushing. A patch stands on an empty floor cell (base z = 1) or on top of a grey step (base z = 2 when the step is at z = 1).
    '["place", [x, y, z]]' for each patch, then '["submit_placement"]'.
- Phase 2 — push: '["push", "+x"]' (or "-x", "+y", "-y") pushes START once in that direction; the whole chain reaction runs and the game is decided at once.
- Win: every gold goal toppled and no red piece toppled.
- Your plan: all placements, then ["submit_placement"], then one push, e.g. [["place", [2, 5, 1]], ["place", [3, 5, 1]], ["submit_placement"], ["push", "+x"]].
"""

PIPE = """- A 3D box with 1-based coordinates (x, y, z); z points up, axes are labelled on the floor edges and the side.
- Portals sit on the side walls: green = entrance, orange = exit. Join every entrance and exit in ONE leak-free pipe network. A portal occupies its own wall cell; it is joined by a pipe in the cell just inside the wall next to it, with one end pointing at the portal.
- Pipes (limited stock, see the PARTS BIN panel): I straight (two opposite ends), L corner (two perpendicular ends), T fork (two opposite ends + one perpendicular). You give a pipe's ends as directions among "+x", "-x", "+y", "-y", "+z", "-z":
    ["place", "L", [x, y, z], ["+x", "-y"]]   ["place", "I", [x, y, z], ["+z", "-z"]]   ["place", "T", [x, y, z], ["+x", "-x", "+z"]]
- Every pipe end must meet the matching end of a neighbouring pipe or a portal (otherwise it leaks).
- Gravity: every pipe must rest on the floor (z = 1), on a support, or on a supported pipe directly below. Supports: ["place", "S1", [x, y, z]] (one cell) or ["place", "S2", [x, y, z]] (two cells: z and z+1). Red crates are obstacles; nothing can be built there.
- ["submit"] checks the network: every portal joined, no leaks, everything supported.
- Your plan: every piece you place, then ["submit"] as the last action.
"""

GAMES = {"type_3": ("FLUIDNET", FLUID, "obs_000_front.png and obs_000_back.png", " (two camera views of the same start state)", '[["open", [2, 3, 1]], ["open", [4, 3, 1]]]'),
         "type_3_1": ("DOMINO CHAIN", DOMINO, "obs_000_front.png", "", '[["place", [2, 5, 1]], ["submit_placement"], ["push", "+x"]]'),
         "type_2_3": ("PIPEFIT", PIPE, "obs_000_front.png and obs_000_back.png", " (two camera views of the same start state)", '[["place", "I", [2, 3, 1], ["+x", "-x"]], ["submit"]]')}

def main(O):
  for line in open(os.path.join(O, "map.txt")):
    sess, _ = line.split()
    num, game = os.path.basename(sess).split("_", 1)
    name, rules, images, views, ex = GAMES[game]
    txt = HEAD.format(name=name, sess=sess, images=images, views=views, cli=CLI, example=ex) + rules + TAIL
    open(os.path.join(O, "prompts", num + ".md"), "w").write(txt)

if __name__ == "__main__":
    main(sys.argv[1])
