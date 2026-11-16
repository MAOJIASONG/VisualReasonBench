"""(1) Can the player take screenshots here?  (2) Which 90-degree rotations does the game accept
at the start of some levels?

    python probe_capture_rotate.py <out_dir> <level> [<level> ...]
"""
import os
import shutil
import sys
from replay import send

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
call = lambda c, p=None: send("127.0.0.1", 9999, 1, c, p, timeout=120)
call("start_level", {"level_index": 1})
call("move", {"piece_id": 5, "axis": "x", "distance": 10, "is_exploratory": False})
r = call("capture")
paths = [p for p in str((r.get("data") or {}).get("screenshot_path") or r.get("screenshot_path") or "").split("|") if p]
print("capture:", r.get("status"), len(paths), "files")
for p in paths:
    ok = os.path.exists(p)
    print("  ", p, "exists" if ok else "MISSING", os.path.getsize(p) if ok else "")
    if ok:
        shutil.copy(p, os.path.join(out, os.path.basename(p)))
for level in map(int, sys.argv[2:]):
    call("start_level", {"level_index": level})
    n = len(call("get_state")["data"]["pieces"])
    ok = []
    for pid in range(n):
        for axis in "xyz":
            for ang in (90, -90):
                call("start_level", {"level_index": level})
                if call("rotate", {"piece_id": pid, "axis": axis, "angle": ang}).get("status") == "success":
                    ok.append(f"{pid}{axis}{ang:+d}")
    print(f"level {level}: {n} pieces, accepted rotations at start: {ok if ok else 'none'}", flush=True)
