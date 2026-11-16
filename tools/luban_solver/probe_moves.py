"""Try every piece x 6 directions at the start of a level (each from a fresh start) and report
which moves the game accepts.

    python probe_moves.py <level> [distance ...]
"""
import sys
import time
from replay import send

level = int(sys.argv[1])
dists = [float(x) for x in sys.argv[2:]] or [1.0, 10.0]
call = lambda c, p=None: send("127.0.0.1", 9999, 1, c, p)
call("start_level", {"level_index": level})
n = len(call("get_state")["data"]["pieces"])
ok = []
for pid in range(n):
    for axis in "xyz":
        for sign in (1, -1):
            for d in dists:
                call("start_level", {"level_index": level})
                r = call("move", {"piece_id": pid, "axis": axis, "distance": sign * d, "is_exploratory": False})
                if r.get("status") == "success":
                    ok.append(f"{pid}{'+' if sign > 0 else '-'}{axis}{d:g}")
print(f"level {level}: {n} pieces, accepted moves: {ok if ok else 'none'}", flush=True)
