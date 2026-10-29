"""
Parallel seed search shared by the CHAIN batch scripts (type_3, type_3_1,
type_2_3).

Every slot of a batch searches its own seed range, lowest seed first.  Jobs
(slot, seed) stream through a process pool: whenever a worker frees up it
gets the next seed of the slot with the fewest seeds in flight, so one slow
seed never holds up the rest.  A slot is finished once the seeds before its
`need`-th candidate have all been evaluated.  Slots are then filled in order,
each taking its lowest-seed candidate whose layout is not already used, so
the picks equal a sequential search over the same seeds (just much faster).

    picks = pick_levels(slots, make, layout_key)

`make(slot, seed)` must be a module-level function returning a result tuple
whose first item is the LevelSpec, or None.  `slots[i]` is passed through
unchanged; slot i searches seeds base + stride * (i + 1) + j, j < n_seeds.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _run(make, slot, seed):
    try:
        return make(slot, seed)
    except Exception as e:                  # one bad seed must not kill the batch
        print(f"  [search] seed {seed}: {type(e).__name__}: {e}", flush=True)
        return None


def _finished(res: Dict[int, Any], n_seeds: int, need: int) -> bool:
    """Seeds 0..j-1 all evaluated and holding `need` candidates, or every
    seed evaluated."""
    found = 0
    for j in range(n_seeds):
        if j not in res:
            return False
        found += res[j] is not None
        if found >= need:
            return True
    return True


def pick_levels(slots: Sequence[Any], make: Callable, key: Callable,
                base: int = 10000, stride: int = 500, n_seeds: int = 300,
                need: int = 3, workers: Optional[int] = None,
                seen: Optional[set] = None) -> List[Optional[Tuple]]:
    """(seed, *result) per slot, or None where no level was found."""
    workers = workers or max(1, min(180, (os.cpu_count() or 2) - 4))
    n = len(slots)
    res: List[Dict[int, Any]] = [{} for _ in range(n)]
    nxt = [0] * n
    flying = [0] * n
    done = [False] * n
    q: "queue.Queue" = queue.Queue()
    t0, last = time.time(), 0.0
    with mp.Pool(workers) as pool:
        in_flight = 0
        while not all(done):
            while in_flight < workers:
                open_ = [i for i in range(n) if not done[i] and nxt[i] < n_seeds]
                if not open_:
                    break
                i = min(open_, key=lambda i: (flying[i], nxt[i]))
                j = nxt[i]
                nxt[i] += 1
                flying[i] += 1
                in_flight += 1
                pool.apply_async(
                    _run, (make, slots[i], base + stride * (i + 1) + j),
                    callback=lambda r, i=i, j=j: q.put((i, j, r)),
                    error_callback=lambda e, i=i, j=j: q.put((i, j, None)))
            if in_flight == 0:
                break
            i, j, r = q.get()
            in_flight -= 1
            flying[i] -= 1
            res[i][j] = r
            if not done[i] and _finished(res[i], n_seeds, need):
                done[i] = True
            if time.time() - last > 30:
                last = time.time()
                print(f"  [search] {sum(done)}/{n} slots done, "
                      f"{sum(len(r) for r in res)} seeds tried, "
                      f"{time.time() - t0:.0f}s", flush=True)
        pool.terminate()                    # drop work for finished slots
    print(f"  [search] {sum(done)}/{n} slots done in {time.time() - t0:.0f}s", flush=True)

    seen = set() if seen is None else seen
    picks: List[Optional[Tuple]] = []
    for i in range(n):
        pick = None
        for j in range(n_seeds):
            if j not in res[i]:
                break                       # only the fully evaluated prefix
            r = res[i][j]
            if r is None:
                continue
            k = key(r[0])
            if k not in seen:
                seen.add(k)
                pick = (base + stride * (i + 1) + j,) + tuple(r)
                break
        picks.append(pick)
    return picks


def tier_table(rows: Sequence[Dict], tier_key: str,
               metrics: Sequence[Tuple[str, str]]) -> List[str]:
    """Markdown lines: min-max of each (key, label) metric per tier, so the
    easy / medium / hard ladder can be checked at a glance."""
    lines = ["| tier | levels | " + " | ".join(lbl for _, lbl in metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for tier in ("easy", "medium", "hard"):
        rs = [r for r in rows if r.get(tier_key) == tier and r.get("seed") is not None]
        cells = []
        for k, _ in metrics:
            v = [float(r[k]) for r in rs if r.get(k) is not None]
            cells.append(f"{min(v):g}–{max(v):g}" if v else "")
        lines.append(f"| {tier} | {len(rs)} | " + " | ".join(cells) + " |")
    return lines
