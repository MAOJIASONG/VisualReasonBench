"""Step-by-step CLI for the Unity Luban (Puzzle) game, for closed-loop agent runs.

    python luban_cli.py start  <level> <session_dir> [--env-id K] [--max-actions 50]
    python luban_cli.py act    <session_dir> '<one JSON action>'
    python luban_cli.py status <session_dir>

The Unity player (Luban/LuBanSuoEnv.x86_64) must be running and listening on
127.0.0.1:$LUBAN_PORT (default 9999); one player serves env_id 1..8 independently.

Actions use the tools of the benchmark's Luban environment (luban_env.get_tool_schemas):
    {"tool":"move_piece","piece_id":2,"axis":"x","distance":30}             (+ optional "is_exploratory":true)
    {"tool":"rotate_piece","piece_id":2,"axis":"y","angle":90}
Every move/rotate counts against the budget, failed ones included. After each action the
observation is the same picture the benchmark gives (cameras 1, 3 and 5 side by side), saved
as obs_<n>.png, plus the piece list from get_state.
"""
import argparse
import json
import os
import socket
import sys
import time

from PIL import Image, ImageDraw, ImageFont

HOST, PORT = "127.0.0.1", int(os.environ.get("LUBAN_PORT", "9999"))
VIEWS = (1, 3, 5)          # same cameras as luban_env.CAPTURE_VIEWS
SETTLE = 0.8               # seconds for the move/rotate animation and the finish-zone trigger


def send(env_id, command, params=None, timeout=120):
    msg = json.dumps({"command": command, "env_id": env_id, "params": params or {}}) + "\n"
    with socket.create_connection((HOST, PORT), timeout=timeout) as s:
        s.sendall(msg.encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.split(b"\n")[0].decode())


def font(size):
    for path in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def capture(env_id, out_path):
    """Tile cameras 1/3/5 side by side with VIEW labels (as luban_env._capture_and_compose)."""
    r = send(env_id, "capture")
    data = r.get("data") or {}
    paths = [p for p in str(data.get("screenshot_path") or r.get("screenshot_path") or "").split("|") if p]
    tiles = []
    for i in VIEWS:
        try:
            tiles.append(Image.open(paths[i]).convert("RGB"))
        except Exception:
            tiles.append(None)
    real = [t for t in tiles if t is not None]
    if not real:
        raise RuntimeError(f"capture failed: {r}")
    h = max(t.height for t in real)
    tiles = [t if t is not None else Image.new("RGB", (real[0].width, h)) for t in tiles]
    tiles = [t if t.height == h else t.resize((round(t.width * h / t.height), h)) for t in tiles]
    gap = 4
    canvas = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), h), (60, 60, 60))
    draw = ImageDraw.Draw(canvas)
    f = font(max(14, h // 24))
    x = 0
    for v, t in zip(VIEWS, tiles):
        canvas.paste(t, (x, 0))
        label = f"VIEW {v}"
        b = draw.textbbox((0, 0), label, font=f)
        draw.rectangle([x + 6, 6, x + 16 + b[2] - b[0], 16 + b[3] - b[1]], fill=(30, 30, 30), outline="white")
        draw.text((x + 11 - b[0], 11 - b[1]), label, fill="white", font=f)
        x += t.width + gap
    canvas.save(out_path)


def describe(data):
    lines = [f"step_count: {data.get('step_count')}, solved: {data.get('is_solved')}",
             f"Pieces ({len(data.get('pieces', []))}):"]
    for p in data.get("pieces", []):
        pos = p.get("pos", {})
        lines.append(f"- id={p['id']}, pos=({pos.get('x')}, {pos.get('y')}, {pos.get('z')}), "
                     f"rot={p.get('rot')}, removed={bool(p.get('is_finished'))}")
    return "\n".join(lines)


def size_hint(level):
    """Range of piece sizes (largest side, logic units) from the game's level file, if available."""
    d = os.environ.get("LUBAN_LEVELS", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "..", "..", "..", "..", "tools", "luban_solver", "levels"))
    try:
        data = json.load(open(os.path.join(d, f"Puzzle_{level}.json")))
        sides = [max(max(v[c] for v in p["vertexes"]) - min(v[c] for v in p["vertexes"]) for c in "xyz")
                 for p in data["polyhedrons"]]
        lo, hi = round(min(sides)), round(max(sides))
        return f"{lo}" if lo == hi else f"{lo}-{hi}"
    except Exception:
        return "40-200"


def load(session):
    return json.load(open(os.path.join(session, "meta.json")))


def save(session, meta):
    tmp = os.path.join(session, ".meta.tmp")
    json.dump(meta, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(session, "meta.json"))


def cmd_start(a):
    os.makedirs(a.session, exist_ok=True)
    send(a.env_id, "start_level", {"level_index": a.level})
    time.sleep(SETTLE)
    st = send(a.env_id, "get_state")["data"]
    capture(a.env_id, os.path.join(a.session, "obs_000.png"))
    meta = {"level": a.level, "env_id": a.env_id, "max_actions": a.max_actions, "size_hint": size_hint(a.level),
            "n_pieces": len(st.get("pieces", [])), "n_actions": 0, "n_failed": 0,
            "status": "playing", "n_finished": 0}
    save(a.session, meta)
    open(os.path.join(a.session, "start.txt"), "w").write(describe(st) + "\n")
    print(describe(st))


def cmd_act(a):
    meta = load(a.session)
    if meta["status"] != "playing":
        print(f"error: the game is over ({meta['status']}); no more actions are accepted.")
        return 1
    try:
        act = json.loads(a.action)
        tool = act["tool"]
        if tool == "move_piece":
            params = {"piece_id": int(act["piece_id"]), "axis": str(act["axis"]).lower(),
                      "distance": float(act["distance"]), "is_exploratory": bool(act.get("is_exploratory", False))}
            if params["axis"] not in ("x", "y", "z"):
                raise ValueError("axis must be x, y or z")
            command = "move"
        elif tool == "rotate_piece":
            params = {"piece_id": int(act["piece_id"]), "axis": str(act["axis"]).lower(),
                      "angle": float(act["angle"])}
            if params["axis"] not in ("x", "y", "z") or params["angle"] % 90 != 0:
                raise ValueError("axis must be x, y or z and angle a multiple of 90")
            command = "rotate"
        else:
            raise ValueError("tool must be move_piece or rotate_piece")
        if not 0 <= params["piece_id"] < meta["n_pieces"]:
            raise ValueError(f"piece_id must be 0..{meta['n_pieces'] - 1}")
    except Exception as e:
        print(f"error: invalid action ({e}). Nothing was played and the action was not counted.")
        return 2

    r = send(meta["env_id"], command, params)
    ok = r.get("status") == "success"
    time.sleep(SETTLE)
    st = send(meta["env_id"], "get_state")["data"]
    meta["n_actions"] += 1
    meta["n_failed"] += 0 if ok else 1
    meta["n_finished"] = sum(bool(p.get("is_finished")) for p in st.get("pieces", []))
    n = meta["n_actions"]
    obs = f"obs_{n:03d}.png"
    capture(meta["env_id"], os.path.join(a.session, obs))
    if st.get("is_solved"):
        meta["status"] = "win"
    elif n >= meta["max_actions"]:
        meta["status"] = "lose"
    save(a.session, meta)
    with open(os.path.join(a.session, "actions.jsonl"), "a") as f:
        f.write(json.dumps({"n": n, "action": act, "result": r.get("status"), "solved": st.get("is_solved"),
                            "n_finished": meta["n_finished"]}) + "\n")
    print(f"result: {'success' if ok else 'failed (blocked: the piece would collide; nothing moved)'}")
    print(f"actions used: {n}/{meta['max_actions']}  status: {meta['status']}")
    print(f"new observation: {obs}")
    print(describe(st))
    return 0


def cmd_status(a):
    print(json.dumps(load(a.session)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("level", type=int); s.add_argument("session")
    s.add_argument("--env-id", type=int, default=1); s.add_argument("--max-actions", type=int, default=50)
    s = sub.add_parser("act"); s.add_argument("session"); s.add_argument("action")
    s = sub.add_parser("status"); s.add_argument("session")
    a = ap.parse_args()
    sys.exit({"start": cmd_start, "act": cmd_act, "status": cmd_status}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
