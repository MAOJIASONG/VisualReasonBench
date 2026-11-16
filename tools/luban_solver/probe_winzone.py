"""Find where a piece counts as finished: move one piece of a level in steps along each
direction and report the first distance at which the game marks it finished.

    python probe_winzone.py <level> <piece_id> [--step 50] [--max 3000]
"""
import argparse
import time
from replay import send

ap = argparse.ArgumentParser()
ap.add_argument("level", type=int)
ap.add_argument("piece", type=int)
ap.add_argument("--step", type=float, default=50)
ap.add_argument("--max", type=float, default=3000)
ap.add_argument("--wait", type=float, default=0.8)
a = ap.parse_args()
call = lambda c, p=None: send("127.0.0.1", 9999, 1, c, p)

for axis in "xyz":
    for sign in (1, -1):
        call("start_level", {"level_index": a.level})
        time.sleep(1.0)
        moved, hit = 0.0, None
        while moved < a.max:
            r = call("move", {"piece_id": a.piece, "axis": axis, "distance": sign * a.step, "is_exploratory": False})
            time.sleep(a.wait)
            st = call("get_state")["data"]
            p = next(q for q in st["pieces"] if q["id"] == a.piece)
            if r.get("status") != "success" or not (r.get("data") or {}).get("pieces"):
                pass
            if p["pos"] == getattr(probe, "last", None) if False else False:
                pass
            if p.get("is_finished"):
                hit = (moved + a.step, p["pos"])
                break
            if r.get("status") != "success":
                break
            moved += a.step
        print(f"level {a.level} piece {a.piece} {'+' if sign > 0 else '-'}{axis}: "
              f"{'finished after ' + str(hit[0]) + ' at ' + str(hit[1]) if hit else 'not finished (moved ' + str(moved) + ', last status ' + r.get('status', '?') + ')'}",
              flush=True)
