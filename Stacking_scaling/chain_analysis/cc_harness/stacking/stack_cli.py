"""Session CLI for the CHAIN Stacking game, for closed-loop CLI agents (Claude Code).

    python stack_cli.py start  <level_index> <session_dir> [--max-actions 30] [--seed N]
    python stack_cli.py act    <session_dir> '<one JSON action>'
    python stack_cli.py status <session_dir>

It reproduces the setting of the chainbench stacking runs (the Kimi-k2.5 rebuttal runs):
the same game code and renderer (game_v1/, copied from CHAIN/assets/stacking_game), the
same puzzles (puzzles_full_v9), the same level indices 0-76, the same tools
(state / place / pickup) and action format, and the same text the agent saw each step
(piece list + last tool result). One action = one step; the budget counts every call,
including failed ones. The episode is won as soon as every cell of the box is filled.

The game state is pickled in <session_dir>/.state.pkl between calls (hidden from the agent).
"""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "game_v1"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from game_core import Vec3  # noqa: E402
from initialization import initialize_pieces_on_ground  # noqa: E402
from loader import create_game_state, load_puzzle_by_name  # noqa: E402
from placement import pickup_piece, place_piece_by_cells  # noqa: E402
from visualizer_3d import visualize_state_3d  # noqa: E402

SS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # Stacking_scaling
PUZZLE_DIR = os.environ.get("STACK_PUZZLE_DIR", os.path.join(SS, "puzzles_full_v9"))
LEVELS = json.load(open(os.path.join(HERE, "levels.json")))


def level_spec(idx: int):
    size, rest = LEVELS[str(idx)].split("_", 1)
    return size, f"puzzle_{rest}", LEVELS[str(idx)]


# ---------------------------------------------------------------- persistence
def _paths(session):
    return os.path.join(session, ".state.pkl"), os.path.join(session, "meta.json")


def load(session):
    pk, mj = _paths(session)
    with open(pk, "rb") as f:
        gs = pickle.load(f)
    return gs, json.load(open(mj))


def save(session, gs, meta):
    pk, mj = _paths(session)
    with open(pk, "wb") as f:
        pickle.dump(gs, f)
    json.dump(meta, open(mj, "w"), indent=1)


def render(session, gs, meta):
    n = meta["n_actions"]
    fig = visualize_state_3d(gs, title=f"{meta['size']}/{meta['puzzle_id']}",
                             show_unplaced=True, figsize=(8, 6))
    name = f"obs_{n:03d}.png"
    fig.savefig(os.path.join(session, name), format="png", dpi=150)
    plt.close(fig)
    return name


# ---------------------------------------------------------------- text the agent sees
def describe_objects(gs, meta):
    """Same text as StackingGameEnvironment.describe_objects (chainbench)."""
    A, B, C = gs.spec.box
    lines = [f"Puzzle: {meta['size']}/{meta['puzzle_id']}",
             f"Box size: {A}x{B}x{C} (1-based grid)",
             f"Placed pieces: {len(gs.placed)}, Unplaced: {len(gs.unplaced)}",
             "Pieces:"]
    for pid in sorted(gs.unplaced):
        lines.append(f" - Piece {pid}: {len(gs.get_piece_def(pid).local_voxels)} voxels (unplaced)")
    for pid, pp in gs.placed.items():
        cs = pp.world_cells
        center = tuple(round(sum(getattr(c, k) for c in cs) / len(cs), 2) for k in "xyz")
        lines.append(f" - Piece {pid}: placed at center {center}, rotation {pp.transform.rot}, "
                     f"cells={len(cs)}")
    if gs.is_complete():
        lines.append("All cells are filled. Puzzle complete.")
    return "\n".join(lines)


def run_tool(gs, act):
    tool = act.get("tool")
    if tool == "state":
        A, B, C = gs.spec.box
        return {"status": "success", "message": "State retrieved",
                "state": {"box": [A, B, C], "occupied": len(gs.occupied), "total_cells": A * B * C,
                          "placed_pieces": sorted(gs.placed), "unplaced_pieces": sorted(gs.unplaced),
                          "is_complete": gs.is_complete()}}
    if tool == "place":
        cells = [Vec3(int(x), int(y), int(z)) for x, y, z in act["cells"]]
        r = place_piece_by_cells(gs, str(act["piece_id"]), cells)
        if r.success:
            return {"status": "success", "message": r.message}
        return {"status": "error", "message": r.message, "error": r.error.value}
    if tool == "pickup":
        r = pickup_piece(gs, str(act["piece_id"]))
        if r.success:
            return {"status": "success", "message": r.message}
        return {"status": "error", "message": r.message, "error": r.error.value}
    return {"status": "error", "message": f"Unknown tool '{tool}'. Use state, place or pickup."}


# ---------------------------------------------------------------- commands
def cmd_start(level, session, max_actions, seed):
    size, pid, name = level_spec(level)
    spec = load_puzzle_by_name(PUZZLE_DIR, size, pid)
    if not spec:
        sys.exit(f"puzzle {size}/{pid} not found under {PUZZLE_DIR}")
    gs = create_game_state(spec)
    initialize_pieces_on_ground(gs, spacing=2, seed=level if seed is None else seed)
    os.makedirs(session, exist_ok=True)
    meta = {"level": level, "name": name, "size": size, "puzzle_id": pid,
            "difficulty": name.split("_")[1], "n_pieces": len(spec.pieces),
            "seed": level if seed is None else seed, "max_actions": max_actions,
            "n_actions": 0, "n_failed": 0, "status": "playing", "log": []}
    img = render(session, gs, meta)
    save(session, gs, meta)
    start = f"Observation: {img}\n\n{describe_objects(gs, meta)}"
    open(os.path.join(session, "start.txt"), "w").write(start + "\n")
    print(f"Started {name}. {start}")


def cmd_act(session, raw):
    gs, meta = load(session)
    if meta["status"] != "playing":
        print(f"The game is over (status: {meta['status']}). No more actions are accepted.")
        return
    try:
        act = json.loads(raw)
        if not isinstance(act, dict):
            raise ValueError("the action must be a JSON object")
        res = run_tool(gs, act)
    except Exception as exc:  # malformed action: counts as a step, like a bad tool call
        act, res = {"raw": raw}, {"status": "error", "message": f"Invalid action: {exc}"}
    meta["n_actions"] += 1
    meta["n_failed"] += res["status"] != "success"
    if gs.is_complete():
        meta["status"] = "win"
    elif meta["n_actions"] >= meta["max_actions"]:
        meta["status"] = "lose"
    meta["log"].append({"n": meta["n_actions"], "action": act, "result": res,
                        "placed": sorted(gs.placed), "complete": gs.is_complete()})
    img = render(session, gs, meta)
    save(session, gs, meta)
    msg = res.get("message", "")
    if "state" in res:
        msg += " " + json.dumps(res["state"])
    print(f"Result: {res['status']} - {msg}")
    print(f"Observation: {img}   actions used {meta['n_actions']}/{meta['max_actions']}")
    print()
    print(describe_objects(gs, meta))
    if meta["status"] == "win":
        print("\nGAME WON: the box is completely filled.")
    elif meta["status"] == "lose":
        print("\nGAME OVER: the action budget is used up.")


def cmd_status(session):
    _, meta = load(session)
    print(json.dumps({k: meta[k] for k in ("level", "name", "difficulty", "n_pieces", "status",
                                           "n_actions", "n_failed", "max_actions")}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("level", type=int); s.add_argument("session")
    s.add_argument("--max-actions", type=int, default=30); s.add_argument("--seed", type=int)
    a = sub.add_parser("act"); a.add_argument("session"); a.add_argument("action")
    t = sub.add_parser("status"); t.add_argument("session")
    o = ap.parse_args()
    if o.cmd == "start":
        cmd_start(o.level, o.session, o.max_actions, o.seed)
    elif o.cmd == "act":
        cmd_act(o.session, o.action)
    else:
        cmd_status(o.session)


if __name__ == "__main__":
    main()
