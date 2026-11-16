"""Single jumps: from the start of a level, move one piece by each distance in one command,
wait, and report whether the game marks it finished.

    python probe_jump.py <level> <piece> <axis> <d1> <d2> ...
"""
import sys
import time
from replay import send

level, piece, axis = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
call = lambda c, p=None: send("127.0.0.1", 9999, 1, c, p)
for d in map(float, sys.argv[4:]):
    call("start_level", {"level_index": level})
    time.sleep(1.0)
    r = call("move", {"piece_id": piece, "axis": axis, "distance": d, "is_exploratory": False})
    out = []
    for w in (0.3, 1.0, 3.0):
        time.sleep(w)
        p = next(q for q in call("get_state")["data"]["pieces"] if q["id"] == piece)
        out.append(p.get("is_finished"))
    print(f"level {level} piece {piece} {axis} {d:+g}: move={r.get('status')} finished after 0.3/1.3/4.3s = {out} pos={p['pos']}", flush=True)
