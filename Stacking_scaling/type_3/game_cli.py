"""
FluidNet 3D - Interactive CLI

Usage:
    python game_cli.py                         # generate random level, play it
    python game_cli.py --load path/to/level.json
    python game_cli.py --seed 17 --size 5 5 5 --targets 2 --valves 5 --protected 2

Commands inside the REPL:
    help                    show command list
    state                   print cell inventory & fluid status
    valves                  list closed valves
    targets                 list target status
    layer <z>               print 2D top-down slice at z
    open <x> <y> <z>        open a valve at (x,y,z)
    render [path.png]       save a 3D render (default /tmp/fluidnet_<turn>.png)
    history                 show valves opened so far
    quit / exit             leave the game
"""

import argparse
import os
import sys
import time
from typing import Optional

# Allow running as `python game_cli.py` from inside the package dir.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_3.game_core import ActionCode, CellType
    from type_3.loader import (
        load_level_from_json, save_level_to_json, create_game_state,
    )
    from type_3.propagation import step_open_valve
    from type_3.generator import generate_level
else:
    from .game_core import ActionCode, CellType
    from .loader import load_level_from_json, save_level_to_json, create_game_state
    from .propagation import step_open_valve
    from .generator import generate_level


COLOR_GLYPHS = {
    "blue":   "\x1b[36m",
    "red":    "\x1b[31m",
    "green":  "\x1b[32m",
    "orange": "\x1b[33m",
    "purple": "\x1b[35m",
    "cyan":   "\x1b[36m",
}
RESET = "\x1b[0m"


def colored(text: str, color: Optional[str]) -> str:
    if not color or color not in COLOR_GLYPHS:
        return text
    return f"{COLOR_GLYPHS[color]}{text}{RESET}"


class FluidNetGame:
    def __init__(self, spec, meta=None):
        self.spec = spec
        self.meta = meta or {}
        self.state = create_game_state(spec)
        self.turn = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def show_header(self):
        A, B, C = self.spec.grid_size
        print(f"\n=== FluidNet 3D  grid {A}x{B}x{C} ===")
        print(f"sources: {len(self.spec.sources)}   "
              f"targets: {len(self.spec.targets)}   "
              f"valves: {len(self.spec.valves)}   "
              f"protected: {len(self.spec.protected)}")
        if self.meta:
            for k, v in self.meta.items():
                print(f"  meta.{k}: {v}")
        print()

    def show_state(self):
        s = self.state
        print(f"\n--- state  turn {self.turn} ---")
        print(f"opened valves: {len(s.opened_valves)}/{len(self.spec.valves)}")
        print(f"fluid cells:   {len(s.fluid_cells)}")
        filled = s.filled_targets()
        print(f"targets filled: {len(filled)}/{len(self.spec.targets)}")
        for t in self.spec.targets:
            tc = self.spec.cells[t].color
            done = s.fluid_cells.get(t) == tc
            tag = colored("FILLED", tc) if done else colored("empty ", tc)
            print(f"   target {t}  color={tc}  [{tag}]")
        if s.terminal:
            print(f"\nGAME OVER — {s.terminal_code.value}")
            if s.violation_cell is not None:
                print(f"  violation at cell {s.violation_cell}")

    def show_valves(self):
        for v in self.spec.valves:
            state = "open " if v in self.state.opened_valves else "CLOSED"
            fluid = self.state.fluid_cells.get(v)
            info = f"  fluid={fluid}" if fluid else ""
            print(f"  valve {v}  [{state}]{info}")

    def show_layer(self, z: int):
        A, B, C = self.spec.grid_size
        if not (1 <= z <= C):
            print(f"z out of range [1..{C}]"); return
        print(f"\n--- z = {z} ---   (y increases upward, x increases right)")
        for y in range(B, 0, -1):
            row = ""
            for x in range(1, A + 1):
                pos = (x, y, z)
                fluid_col = self.state.fluid_cells.get(pos)
                cell = self.spec.cells.get(pos)
                if cell is None:
                    row += " . "
                    continue
                if cell.type == CellType.SOURCE:
                    ch = colored(" S ", cell.color)
                elif cell.type == CellType.TARGET:
                    ch = colored(" T ", cell.color) if fluid_col else colored(" t ", cell.color)
                elif cell.type == CellType.VALVE:
                    opened = pos in self.state.opened_valves
                    ch = " v " if opened else " V "
                elif cell.type == CellType.PROTECTED:
                    ch = " X "
                elif cell.type == CellType.PIPE:
                    ch = colored(" o ", fluid_col) if fluid_col else " . "
                else:
                    ch = " . "
                row += ch
            print(f"  y={y:2d} | {row}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def cmd_open(self, x: int, y: int, z: int):
        if self.state.terminal:
            print("game is terminal; reload to replay")
            return
        result = step_open_valve(self.state, (x, y, z))
        self.turn += 1 if result.success else 0
        print(f"  -> {result.code.value}: {result.message}")
        if result.new_fluid:
            print(f"  fluid propagated to {len(result.new_fluid)} new cells")
        if result.newly_filled:
            print(f"  newly filled targets: {sorted(result.newly_filled)}")
        if result.code == ActionCode.WIN:
            print("\n🎉 ALL TARGETS FILLED — YOU WIN!")
        elif result.code == ActionCode.NO_PROGRESS_DEADLOCK:
            print("\n💀 No remaining valve can progress.  Puzzle lost.")
        elif self.state.terminal:
            print(f"\n💥 Game over ({result.code.value})")

    def cmd_render(self, save_path: Optional[str] = None):
        from .visualizer_3d import render_state
        if save_path is None:
            ts = time.strftime("%H%M%S")
            save_path = f"/tmp/fluidnet_t{self.turn:02d}_{ts}.png"
        render_state(self.state, save_path=save_path)
        print(f"  saved render to {save_path}")

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def show_help(self):
        print("""
commands:
  help                       show this message
  state                      show fluid / target status
  valves                     list valves with open/closed state
  targets                    alias of state, target-only view
  layer <z>                  print 2D slice at height z (1-based)
  open <x> <y> <z>           open a valve at cell (x,y,z)
  render [path.png]          save a 3D render
  history                    list valve opens in order
  quit / exit                leave
""")

    def run(self):
        self.show_header()
        self.show_valves()
        while True:
            try:
                line = input("\nfluidnet> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            try:
                if cmd in ("quit", "exit"):
                    return
                elif cmd == "help":
                    self.show_help()
                elif cmd == "state":
                    self.show_state()
                elif cmd == "valves":
                    self.show_valves()
                elif cmd == "targets":
                    self.show_state()
                elif cmd == "layer":
                    self.show_layer(int(parts[1]))
                elif cmd == "open":
                    x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                    self.cmd_open(x, y, z)
                elif cmd == "render":
                    self.cmd_render(parts[1] if len(parts) >= 2 else None)
                elif cmd == "history":
                    for i, v in enumerate(self.state.move_history):
                        print(f"  {i+1}. open {v}")
                else:
                    print(f"unknown command: {cmd!r}.  type 'help'")
            except (ValueError, IndexError) as e:
                print(f"  error: {e}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=None, help="load level JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, nargs=3, default=[5, 5, 5],
                    metavar=("A", "B", "C"))
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--valves", type=int, default=5)
    ap.add_argument("--protected", type=int, default=2)
    ap.add_argument("--save", type=str, default=None,
                    help="if generating, save the level JSON to this path")
    args = ap.parse_args()

    if args.load:
        spec = load_level_from_json(args.load)
        meta = {"loaded_from": args.load}
    else:
        result = generate_level(
            grid_size=tuple(args.size),
            n_targets=args.targets,
            n_valves=args.valves,
            n_protected=args.protected,
            seed=args.seed,
        )
        if result is None:
            print("failed to generate a valid level (try a different seed)")
            return 1
        spec, meta = result
        if args.save:
            save_level_to_json(spec, args.save, meta=meta)
            print(f"saved generated level to {args.save}")

    game = FluidNetGame(spec, meta=meta)
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
