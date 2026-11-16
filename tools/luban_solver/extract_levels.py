"""Extract the 32 level files (Puzzle_0 .. Puzzle_31) from the Unity build of the Luban game.

    python extract_levels.py <LuBanSuoEnv_Data/resources.assets> <out_dir>

Each level is stored in resources.assets as a TextAsset: name, 4-byte length, JSON text.
"""
import json
import os
import re
import struct
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
data = open(src, "rb").read()
n = 0
for m in re.finditer(rb"(Puzzle_\d+)\x00{0,3}(....)\{\s*\"polyhedrons\"", data, re.S):
    name = m.group(1).decode()
    size = struct.unpack("<I", m.group(2))[0]
    start = m.start(2) + 4
    text = data[start:start + size].decode("utf-8")
    json.loads(text)  # must parse
    open(os.path.join(out, name + ".json"), "w").write(text)
    n += 1
print(f"{n} levels -> {out}")
