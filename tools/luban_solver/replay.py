"""Replay a solver result in the Unity Luban game and check that the game counts it as solved.

    python replay.py <result.json> [--env-id 1] [--host 127.0.0.1] [--port 9999]

The Unity player (Luban/LuBanSuoEnv.x86_64) must already be running, e.g. under xvfb-run on
the cluster (see scripts/linux/run_luban_linux.sh). Sends start_level, then every move of the
solution as a `move` command, and prints the game's answer after each one.
"""
import argparse
import json
import socket
import time
import sys


def send(host, port, env_id, command, params=None, timeout=60):
    msg = json.dumps({"command": command, "env_id": env_id, "params": params or {}}) + "\n"
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(msg.encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.split(b"\n")[0].decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--env-id", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    a = ap.parse_args()

    res = json.load(open(a.result))
    if res["status"] != "solved":
        sys.exit(f"{res['level']}: no solution to replay ({res['status']})")
    level = int(res["level"].split("_")[1])
    call = lambda c, p=None: send(a.host, a.port, a.env_id, c, p)

    r = call("start_level", {"level_index": level})
    print("start_level:", r.get("status"))
    for k, m in enumerate(res["moves"], 1):
        r = call("move", {"piece_id": m["piece_id"], "axis": m["axis"],
                          "distance": m["distance"], "is_exploratory": False})
        time.sleep(0.8)   # let the move animation and the finish-zone trigger run
        st = call("get_state").get("data", {})
        finished = [p["id"] for p in st.get("pieces", []) if p.get("is_finished")]
        print(f"{k:2d} move piece {m['piece_id']} {m['axis']} {m['distance']:+g}: {r.get('status')} "
              f"| finished={finished} solved={st.get('is_solved')}"
              f"{'' if m.get('finishes_piece') == (m['piece_id'] in finished) else '  <-- differs from solver'}")
    st = call("get_state").get("data", {})
    print(f"RESULT {res['level']}: is_solved={st.get('is_solved')} after {len(res['moves'])} moves "
          f"(game step_count={st.get('step_count')})")


if __name__ == "__main__":
    main()
