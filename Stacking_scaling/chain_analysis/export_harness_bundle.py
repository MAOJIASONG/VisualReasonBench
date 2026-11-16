"""Bundle the agent-harness results of Experiments 1, 2 and 6 (seed 1, base + plan prompts):
an Excel workbook (one sheet per experiment), one folder of output files per experiment, and
a README.  Raw images and full agent transcripts are left out (tens of GB); they stay in cc_runs/.

    cd Stacking_scaling && python chain_analysis/export_harness_bundle.py --out ../harness_bundle
"""
import argparse, csv, glob, json, os, shutil, statistics as stx, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SS = os.path.abspath(os.path.join(HERE, ".."))
CC = os.path.abspath(os.path.join(SS, "..", "cc_runs"))
PAPER = os.path.abspath(os.path.join(SS, "..", "ICLR26_CHAIN"))
sys.path.insert(0, SS)

from chain_analysis.envs import load_env                                  # noqa: E402
from chain_analysis.stacking_oracle.exp2_stacking import success_curve    # noqa: E402

# agent display name, harness, short key used in Exp 1/2 files, Claude alias / Codex model id
AGENTS = [("Claude-Opus-5.5", "Claude Code", "Opus", "opus"),
          ("Claude-Sonnet-5.5", "Claude Code", "Sonnet", "sonnet"),
          ("GPT-6.1-sol", "Codex CLI", "sol", "gpt-6.1-sol"),
          ("GPT-6.1-terra", "Codex CLI", "terra", "gpt-6.1-terra"),
          ("GPT-6-luna", "Codex CLI", "luna", "gpt-6-luna")]
VARIANTS = ["base", "plan"]
GAMES = [("type_3", "FluidNet"), ("type_2_3", "PipeFit"), ("type_3_1", "Domino")]
GNAME = dict(GAMES)
FAMILIES = ["Puzzle", "Stacking", "FluidNet", "PipeFit", "Domino"]
EP_KEEP = {"agent_calls.log", "episode.log", "prompt.md", "meta.json", "actions.jsonl", "start.txt",
           ".state.json"}
RUN_KEEP = ("summary.csv", "audit.json", "results_*.jsonl", "traj_*.jsonl")


def run_dir(fam, key, variant):
    codex = key.startswith("gpt")
    if fam == "hard45":
        return f"hard45_codex_{variant}_{key}" if codex else f"hard45_{key}_{variant}"
    if fam == "stack77":
        if codex:
            return f"stack77_codex_{key}" if variant == "base" else f"stack77_codex_plan_{key}"
        return f"stack77_{key}_{variant}"
    pre = "luban32" if variant == "base" else "luban32plan"
    return f"luban32/{pre}_{'codex' if codex else 'claude'}_{key}"


def rd_csv(p):
    return list(csv.DictReader(open(p)))


def total_tokens(r):
    """Paper's token count: Claude = input + output + cache read + cache creation;
    Codex = input (cached input is part of it) + output (reasoning is part of it)."""
    if "cache_read_input_tokens" in r:
        return sum(int(r[c] or 0) for c in ("input_tokens", "output_tokens",
                                            "cache_read_input_tokens", "cache_creation_input_tokens"))
    return int(r["input_tokens"] or 0) + int(r["output_tokens"] or 0)


_opt = {}


def opt_len(game, level):
    if (game, level) not in _opt:
        _opt[(game, level)] = len(load_env(os.path.join(game, "puzzles_50", level)).reference_plan())
    return _opt[(game, level)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return stx.mean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return stx.median(xs) if xs else None


def r2(x, n=2):
    return None if x is None else round(x, n)


# ---------------------------------------------------------------------------------------------
# Experiment 6: harness runs
# ---------------------------------------------------------------------------------------------

def exp6_episodes():
    """One row per episode over Puzzle / Stacking / three games, both prompt variants."""
    rows = []
    for name, harness, short, key in AGENTS:
        for v in VARIANTS:
            for r in rd_csv(os.path.join(CC, run_dir("luban32", key, v), "summary.csv")):
                rows.append(dict(agent=name, harness=harness, prompt=v, family="Puzzle", run_dir=run_dir("luban32", key, v),
                                 episode=r["episode"], level=r["level"], difficulty="", n_pieces=r["n_pieces"],
                                 status=r["status"], win=int(r["status"] == "win"), actions_used=int(r["actions_used"]),
                                 rejected_actions=r.get("failed_actions"), verifies="", optimal_len=1, raw=r))
            for r in rd_csv(os.path.join(CC, run_dir("stack77", key, v), "summary.csv")):
                rows.append(dict(agent=name, harness=harness, prompt=v, family="Stacking", run_dir=run_dir("stack77", key, v),
                                 episode=r["episode"], level=r["level"], difficulty=r["difficulty"], n_pieces=r["n_pieces"],
                                 status=r["status"], win=int(r["status"] == "win"), actions_used=int(r["actions_used"]),
                                 rejected_actions=r.get("failed_actions"), verifies="", optimal_len=int(r["n_pieces"]), raw=r))
            d = run_dir("hard45", key, v)
            ver = {}
            for g, _ in GAMES:
                for l in open(os.path.join(CC, d, f"results_{g}.jsonl")):
                    if l.strip():
                        x = json.loads(l)
                        ver[f"{g}__{x['level']}"] = x
            for r in rd_csv(os.path.join(CC, d, "summary.csv")):
                g, lvl = r["episode"].split("__", 1)
                x = ver.get(r["episode"], {})
                rows.append(dict(agent=name, harness=harness, prompt=v, family=GNAME[g], run_dir=d,
                                 episode=r["episode"], level=lvl, difficulty="hard (recalibrated)", n_pieces="",
                                 status=r["status"], win=int(r["status"] == "win"), actions_used=int(r["actions_used"]),
                                 rejected_actions=x.get("rejected"), verifies=x.get("verifies"),
                                 optimal_len=opt_len(g, lvl), raw=r))
    for e in rows:
        r = e.pop("raw")
        e["tool_calls"] = r.get("tool_calls")
        e["launches"] = r.get("launches")
        e["api_errors"] = r.get("api_errors")
        e["duration_min"] = r.get("duration_min")
        e["cost_usd_claude"] = r.get("cost_usd") if e["harness"] == "Claude Code" else ""
        for c in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                  "cached_input_tokens", "reasoning_output_tokens"):
            e[c] = r.get(c, "")
        e["total_tokens"] = total_tokens(r)
        e["flagged_calls"] = r.get("flagged_calls")
        o = e["optimal_len"]
        e["norm_dist"] = round(max(0, e["actions_used"] - o) / max(1, o), 4) if e["win"] else ""
    return rows


def exp6_summary(eps):
    out = []
    for name, harness, short, key in AGENTS:
        for v in VARIANTS:
            es = [e for e in eps if e["agent"] == name and e["prompt"] == v]
            fam = {f: [sum(e["win"] for e in es if e["family"] == f), sum(e["family"] == f for e in es)]
                   for f in FAMILIES}
            hs = [e for e in es if e["family"] == "Stacking" and e["difficulty"] == "hard"]
            won = [e for e in es if e["win"]]
            out.append(dict(agent=name, harness=harness, prompt=v, fam=fam,
                            stack_hard=[sum(e["win"] for e in hs), len(hs)],
                            steps=mean(e["actions_used"] for e in won),
                            ndist=mean(e["norm_dist"] for e in won),
                            tokens_m=sum(e["total_tokens"] for e in es) / 1e6,
                            stopped=sum(e["status"] == "playing" for e in es)))
    return out


# ---------------------------------------------------------------------------------------------
# Experiment 1: fatal steps
# ---------------------------------------------------------------------------------------------

def exp1_stacking():
    eps = [json.loads(l) for l in open(os.path.join(HERE, "stacking_oracle/results/exp1_cc/episodes.jsonl"))]
    out = []
    for name, harness, short, key in AGENTS:
        for v in VARIANTS:
            for scope in ("all 77", "hard 47"):
                es = [e for e in eps if e["model"] == f"{short} {v}" and (scope == "all 77" or e["difficulty"] == "hard")]
                f = [e for e in es if e["fatal_step"] is not None]
                lost = [e for e in es if not e["win"]]
                out.append(dict(
                    agent=name, prompt=v, scope=scope, episodes=len(es), won=sum(e["win"] for e in es),
                    fatal=len(f), fatal_mean=r2(mean(e["fatal_step"] for e in f)), fatal_median=median(e["fatal_step"] for e in f),
                    delay_mean=r2(mean(e["delay"] for e in f)), delay_median=median(e["delay"] for e in f),
                    never_exposed=sum(e["exposure_step"] is None and e["repair_step"] is None for e in f),
                    repaired=sum(e["repair_step"] is not None for e in f),
                    dead_share=r2(mean(e["dead_share"] for e in es), 4),
                    failed=len(lost), failed_dead=sum(e["ended_dead"] for e in lost),
                    failed_dead_hidden=sum(e["ended_dead_hidden"] for e in lost),
                    actions=sum(e["n_actions"] for e in es), rejected=sum(e["rejected"] for e in es)))
    return eps, out


def exp1_games():
    out, flat = [], []
    for g, gname in GAMES:
        steps = [json.loads(l) for l in open(os.path.join(HERE, f"exp1_vs_agents/{g}/steps.jsonl"))]
        for s in steps:
            flat.append({"game": gname, **{k: v for k, v in s.items() if k != "steps"}})
        for name, harness, short, key in AGENTS:
            for v in VARIANTS:
                es = [s for s in steps if s["model"] == f"{short} {v}"]
                f = [s for s in es if s["fatal_step"] is not None]
                out.append(dict(
                    game=gname, agent=name, prompt=v, runs=len(es), won=sum(bool(s["win"]) for s in es), fatal=len(f),
                    fatal_mean=r2(mean(s["fatal_step"] for s in f)), fatal_median=median(s["fatal_step"] for s in f),
                    delay_mean=r2(mean(s["consequence_delay"] for s in f)),
                    delay_median=median(s["consequence_delay"] for s in f),
                    never_exposed=sum(s["exposed_step"] is None for s in f),
                    hidden_at_end=sum(s["exposed_step"] is None and s["repaired_step"] is None for s in f),
                    repaired=sum(s["repaired_step"] is not None for s in f),
                    mean_V=r2(mean(s["mean_V"] for s in es), 3), V_minus_rand=r2(mean(s["mean_V_minus_rand"] for s in es), 3),
                    wasted=r2(mean(s["wasted"] for s in es)), illegal=sum(s["illegal"] or 0 for s in es)))
    return flat, out


def fatal_curves(eps, T=30):
    """Figure data: % of hard Stacking runs not yet in a dead state / with no failure exposed, by action."""
    rows = []
    for name, harness, short, key in AGENTS:
        es = [e for e in eps if e["model"] == f"{short} base" and e["difficulty"] == "hard"]
        fat = [e["fatal_step"] for e in es]
        exp = [e["exposure_step"] if e["fatal_step"] else None for e in es]
        for kind, vals in (("not yet dead", fat), ("no failure exposed", exp)):
            rows.append([name, kind] + [round(100 * sum(x is None or x > t for x in vals) / len(vals), 1)
                                        for t in range(T + 1)])
    return rows


# ---------------------------------------------------------------------------------------------
# Experiment 2: budget-matched search
# ---------------------------------------------------------------------------------------------

S_BUDGETS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
S_METHODS = [("random", "random"), ("greedy", "greedy"), ("greedy_undo", "greedy + undo"), ("dfs_prune", "DFS + pruning")]
G_BUDGETS = [1, 2, 3, 5, 10, 20, 64]
G_METHODS = [("random", "random"), ("greedy", "greedy"), ("greedy_undo", "greedy + undo"), ("dfs", "DFS"), ("bfs", "BFS")]


def exp2_stacking():
    runs = [json.loads(l) for l in open(os.path.join(HERE, "stacking_oracle/results/runs.jsonl"))]
    traj = [json.loads(l) for l in open(os.path.join(HERE, "stacking_oracle/results/traj_cc.jsonl"))]
    names = sorted({r["level"] for r in runs})
    scopes = {"all 77": names, "hard 47": [n for n in names if "_hard_" in n]}
    curves, need = [], []
    for scope, lv in scopes.items():
        for m, mname in S_METHODS:
            c = success_curve([r for r in runs if r["method"] == m], lv, S_BUDGETS)
            curves.append([scope, "search", mname] + [round(100 * c[b], 1) for b in S_BUDGETS])
        for name, harness, short, key in AGENTS:
            for v in VARIANTS:
                mine = {e["level"]: e for e in traj if e["model"] == f"{short} {v}"}
                curves.append([scope, f"agent ({v})", name] + [
                    round(100 * sum(1 for l in lv if mine[l]["win"] and mine[l]["n_actions"] <= b) / len(lv), 1)
                    if b <= 30 else None for b in S_BUDGETS])
    # actions each search method needs to match an agent's success within its 30 actions
    allv = scopes["all 77"]
    caps = {m: [r for r in runs if r["method"] == m] for m, _ in S_METHODS}
    # success per budget 1..1000, computed once per method
    full = {m: success_curve(caps[m], allv, range(1, 1001)) for m, _ in S_METHODS}
    for name, harness, short, key in AGENTS:
        for v in VARIANTS:
            mine = {e["level"]: e for e in traj if e["model"] == f"{short} {v}"}
            target = sum(mine[l]["win"] for l in allv) / len(allv)
            row = [name, v, round(100 * target, 1)]
            for m, _ in S_METHODS:
                b = next((b for b in range(1, 1001) if full[m][b] >= target - 1e-9), None)
                row.append(b if b is not None else ">1000")
            need.append(row)
    return curves, need


def exp2_games():
    hard = {g: {l.strip().split("/")[-1] for l in open(os.path.join(CC, "hard45_levels.txt")) if l.startswith(g + "/")}
            for g, _ in GAMES}
    base, agent = {}, {}
    for g, _ in GAMES:
        for l in open(os.path.join(SS, g, "analysis_exp2", "runs.jsonl")):
            r = json.loads(l)
            if r["level"] in hard[g]:
                base.setdefault((g, r["method"]), []).append({"win": r["solved"], "verifies": r["verifies"]})
        for name, harness, short, key in AGENTS:
            for v in VARIANTS:
                last = {}
                for l in open(os.path.join(CC, run_dir("hard45", key, v), f"results_{g}.jsonl")):
                    if l.strip():
                        r = json.loads(l)
                        if r["level"] in hard[g]:
                            last[r["level"]] = r
                agent[(g, name, v)] = list(last.values())
    rate = lambda rs, b: round(100 * sum(r["win"] and r["verifies"] <= b for r in rs) / len(rs), 1) if rs else None
    rows = []
    for g, gname in GAMES + [("all", "All 45")]:
        pick = lambda src, k: src.get((g,) + k, []) if g != "all" else [r for gg, _ in GAMES for r in src.get((gg,) + k, [])]
        for name, harness, short, key in AGENTS:
            for v in VARIANTS:
                rows.append([gname, f"agent ({v})", name] + [rate(pick(agent, (name, v)), b) for b in G_BUDGETS])
        for m, mname in G_METHODS:
            rows.append([gname, "search", mname] + [rate(pick(base, (m,)), b) for b in G_BUDGETS])
    return rows


# ---------------------------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------------------------

from openpyxl import Workbook                                              # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side     # noqa: E402
from openpyxl.utils import get_column_letter as COL                        # noqa: E402

FONT = "Arial"
TITLE_F = Font(name=FONT, bold=True, size=12)
NOTE_F = Font(name=FONT, italic=True, size=9, color="595959")
HEAD_F = Font(name=FONT, bold=True, size=10, color="FFFFFF")
BODY_F = Font(name=FONT, size=10)
HEAD_FILL = PatternFill("solid", fgColor="2F5496")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
PCT, D1, D2, D3 = "0.0%", "0.0", "0.00", "0.000"


def block(ws, r, title, notes, headers, rows, fmts=None):
    """Write a titled table at row r; cells given as callables get the sheet row number
    (to build formulas).  Returns the first free row after the table."""
    ws.cell(r, 1, title).font = TITLE_F
    for n in notes:
        r += 1
        ws.cell(r, 1, n).font = NOTE_F
    r += 1
    for j, h in enumerate(headers, 1):
        c = ws.cell(r, j, h)
        c.font, c.fill, c.border = HEAD_F, HEAD_FILL, BOX
        c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.row_dimensions[r].height = 45
    top = r + 1
    for i, row in enumerate(rows):
        for j, val in enumerate(row, 1):
            if callable(val):
                val = val(top + i)
            c = ws.cell(top + i, j, val)
            c.font, c.border = BODY_F, BOX
            if fmts and j in fmts:
                c.number_format = fmts[j]
    return top + len(rows) + 2


def ratio(num_col, den_col):
    return lambda r: f'=IF({COL(den_col)}{r}=0,"",{COL(num_col)}{r}/{COL(den_col)}{r})'


def sheet_exp6(ws, s6, eps):
    fams = ["Puzzle", "Stacking", "Stacking (hard)", "FluidNet", "PipeFit", "Domino"]
    heads = ["Agent", "Harness", "Prompt"]
    for f in fams:
        heads += [f"{f} won", f"{f} episodes", f"{f} success"]
    heads += ["Pass@1 (macro over 5 families)", "Succ. (solved, 5 families)", "Steps (mean actions, solved)",
              "NormDist (mean, solved)", "Total tokens (M)", "Solv./Tok. (solved per 1M tokens)",
              "Stopped early (status = playing)"]
    rows, fm = [], {}
    for s in s6:
        vals = [s["fam"]["Puzzle"], s["fam"]["Stacking"], s["stack_hard"], s["fam"]["FluidNet"],
                s["fam"]["PipeFit"], s["fam"]["Domino"]]
        row = [s["agent"], s["harness"], s["prompt"]]
        for k, (w, n) in enumerate(vals):
            row += [w, n, ratio(4 + 3 * k, 5 + 3 * k)]
            fm[6 + 3 * k] = PCT
        rate_cols = [6, 9, 15, 18, 21]          # Puzzle, Stacking, FluidNet, PipeFit, Domino
        won_cols = [4, 7, 13, 16, 19]
        row += [lambda r, c=rate_cols: "=AVERAGE(" + ",".join(f"{COL(x)}{r}" for x in c) + ")",
                lambda r, c=won_cols: "=" + "+".join(f"{COL(x)}{r}" for x in c),
                r2(s["steps"]), r2(s["ndist"], 3), r2(s["tokens_m"]), ratio(23, 26), s["stopped"]]
        rows.append(row)
    fm.update({22: PCT, 24: D1, 25: D2, 26: D1, 27: D2})
    r = block(ws, 1, "Experiment 6 — agent harnesses (Claude Code / Codex CLI): main results, seed 1",
              ["Puzzle = Luban lock, 32 levels, 50 actions; Stacking = 77 levels (10 easy / 20 mid / 47 hard), 30 actions; "
               "FluidNet / PipeFit / Domino = 15 recalibrated hard levels each, 40 actions. One episode per level. "
               "Codex reasoning effort = high.",
               "Base rows = the harness rows of tab:main-results in the paper (rounded there). Success, Pass@1, Succ. and "
               "Solv./Tok. are formulas; Steps, NormDist and tokens are computed from the per-episode file "
               "exp6/exp6_episodes.csv. Pass@1 does not include the Stacking (hard) column."],
              heads, rows, fm)
    # detail by family
    heads2 = ["Agent", "Prompt", "Family", "Episodes", "Won", "Lost", "Stopped early", "Success",
              "Mean actions (all episodes)", "Mean actions (solved)", "Mean rejected actions",
              "Mean verifications (three games)", "Mean tool calls", "Total duration (h)", "Total tokens (M)",
              "Claude cost (USD)"]
    rows2 = []
    for name, harness, short, key in AGENTS:
        for v in VARIANTS:
            for f in FAMILIES:
                es = [e for e in eps if e["agent"] == name and e["prompt"] == v and e["family"] == f]
                rej = [float(e["rejected_actions"]) for e in es if e["rejected_actions"] not in (None, "")]
                ver = [float(e["verifies"]) for e in es if e["verifies"] not in (None, "")]
                cost = sum(float(e["cost_usd_claude"] or 0) for e in es) if harness == "Claude Code" else None
                rows2.append([name, v, f, len(es), sum(e["win"] for e in es), sum(e["status"] == "lose" for e in es),
                              sum(e["status"] == "playing" for e in es), ratio(5, 4),
                              r2(mean(e["actions_used"] for e in es)),
                              r2(mean(e["actions_used"] for e in es if e["win"])),
                              r2(mean(rej)), r2(mean(ver)) if f in GNAME.values() else None,
                              r2(mean(float(e["tool_calls"] or 0) for e in es)),
                              r2(sum(float(e["duration_min"] or 0) for e in es) / 60),
                              r2(sum(e["total_tokens"] for e in es) / 1e6),
                              r2(cost) if cost is not None else None])
    block(ws, r, "Experiment 6 — detail by agent, prompt and family",
          ["Stopped early = the agent ended its session before winning or losing (budget left). "
           "Verifications only exist in FluidNet / PipeFit / Domino. Claude cost is the cost reported by Claude Code; "
           "Codex ran on a ChatGPT login and reports no cost."],
          heads2, rows2, {8: PCT, 9: D1, 10: D1, 11: D1, 12: D1, 13: D1, 14: D1, 15: D1, 16: "$#,##0.00"})
    ws.column_dimensions["A"].width = 19
    for j in range(2, 30):
        ws.column_dimensions[COL(j)].width = 11
    ws.freeze_panes = "B1"


def sheet_exp1(ws, st, gm, curves):
    heads = ["Agent", "Prompt", "Levels", "Episodes", "Won", "Win rate", "Runs with a fatal step", "Fatal rate",
             "Fatal step (mean)", "Fatal step (median)", "Delay to exposure (mean)", "Delay to exposure (median)",
             "Never exposed", "Repaired", "Repaired (% of fatal runs)", "Share of actions in dead states",
             "Failed runs", "Failed runs ending dead", "Failed: ending dead (%)", "Failed: ending dead, still hidden",
             "Actions", "Rejected actions", "Rejected (% of actions)"]
    rows = []
    for s in st:
        rows.append([s["agent"], s["prompt"], s["scope"], s["episodes"], s["won"], ratio(5, 4), s["fatal"], ratio(7, 4),
                     s["fatal_mean"], s["fatal_median"], s["delay_mean"], s["delay_median"], s["never_exposed"],
                     s["repaired"], ratio(14, 7), s["dead_share"], s["failed"], s["failed_dead"], ratio(18, 17),
                     s["failed_dead_hidden"], s["actions"], s["rejected"], ratio(22, 21)])
    r = block(ws, 1, "Experiment 1 — fatal steps on Stacking (exact-cover oracle), agent harness runs, seed 1",
              ["Dead state = the placed pieces fit no completion (exact cover of the free cells by the unplaced pieces). "
               "Fatal step = index of the first accepted action that enters a dead state. Exposure = first later action at "
               "which the dead state is visible (a remaining piece fits nowhere, or a free cell cannot be covered).",
               "Delay = exposure − fatal step (runs never exposed are left out of the mean/median). Repaired = the agent "
               "left the dead state again with pickups. Fatal-step and delay statistics are over runs with a fatal step. "
               "The paper quotes the 'hard 47' rows, base prompt (means)."],
              heads, rows, {6: PCT, 8: PCT, 9: D2, 10: D1, 11: D2, 12: D1, 15: PCT, 16: PCT, 19: PCT, 23: PCT})
    heads2 = ["Game", "Agent", "Prompt", "Runs", "Won", "Win rate", "Runs with a fatal step", "Fatal rate",
              "Fatal step (mean)", "Fatal step (median)", "Delay to exposure (mean)", "Delay to exposure (median)",
              "Fatal, never exposed", "Fatal, hidden at end (never exposed or repaired)", "Repaired",
              "Repaired (% of fatal runs)", "Mean V (solution retention per step)", "Mean V − V_rand",
              "Mean wasted steps", "Illegal actions"]
    rows2 = [[s["game"], s["agent"], s["prompt"], s["runs"], s["won"], ratio(5, 4), s["fatal"], ratio(7, 4),
              s["fatal_mean"], s["fatal_median"], s["delay_mean"], s["delay_median"], s["never_exposed"],
              s["hidden_at_end"], s["repaired"], ratio(15, 7), s["mean_V"], s["V_minus_rand"], s["wasted"], s["illegal"]]
             for s in gm]
    r = block(ws, r, "Experiment 1 — fatal steps on FluidNet / PipeFit / Domino (enumeration oracle), 15 hard levels each",
              ["Dead end = N(s') = 0 winning end states reachable without a reset. Exposure = first failed verification "
               "(valve touching a red cell, failed submit, losing push). V = N(s')/N(s) per step; V_rand = mean V over the "
               "legal progress actions in the same state. Not in the paper's main text (removed; Stacking only)."],
              heads2, rows2, {6: PCT, 8: PCT, 9: D2, 10: D1, 11: D2, 12: D1, 16: PCT, 17: D3, 18: D3, 19: D2})
    heads3 = ["Agent", "Curve"] + [f"t={t}" for t in range(31)]
    block(ws, r, "Figure data — fig:fatal-curves (left, middle): % of hard Stacking runs (47, base prompt) by action t",
          ["'not yet dead' = no fatal step at or before action t; 'no failure exposed' = the dead state has not "
           "become visible at or before action t."],
          heads3, curves, {j: D1 for j in range(3, 34)})
    ws.column_dimensions["A"].width = 19
    for j in range(2, 34):
        ws.column_dimensions[COL(j)].width = 11
    ws.freeze_panes = "B1"


def sheet_exp2(ws, curves, need, games):
    heads = ["Levels", "Type", "Method / agent"] + [f"≤{b} actions" for b in S_BUDGETS]
    r = block(ws, 1, "Experiment 2 — Stacking: success (%) within an action budget, search baselines vs agents",
              ["Search baselines know the exact rules, see the true state and never make an illegal move; 5 seeds each, "
               "mean over seeds. Agents had a 30-action budget, so their columns stop at ≤30. Agent counts include "
               "rejected actions. 'DFS + pruning' is the 'best search' of fig:fatal-curves (right)."],
              heads, curves, {j: D1 for j in range(4, 4 + len(S_BUDGETS))})
    heads2 = ["Agent", "Prompt", "Agent success, all 77 (%)"] + [f"Actions {m} needs to match" for _, m in S_METHODS]
    r = block(ws, r, "Experiment 2 — Stacking: smallest action budget at which each search method reaches the agent's success",
              ["Agent success = share of all 77 levels won within its 30 actions. '>1000' = never reached within the cap."],
              heads2, need, {3: D1})
    heads3 = ["Game", "Type", "Method / agent"] + [f"≤{b} verifications" for b in G_BUDGETS]
    block(ws, r, "Experiment 2 — FluidNet / PipeFit / Domino (15 hard levels each): success (%) within a verification budget",
          ["Verification = an action that can end the game (opening a valve, submit, push). Search baselines: one run per "
           "seed (random, greedy, DFS: 5 seeds; greedy + undo, BFS: deterministic); agents: one episode per level, "
           "40-action budget. Not in the paper's main text (removed; Stacking only)."],
          heads3, games, {j: D1 for j in range(4, 4 + len(G_BUDGETS))})
    ws.column_dimensions["A"].width = 19
    ws.column_dimensions["C"].width = 19
    for j in range(4, 20):
        ws.column_dimensions[COL(j)].width = 11
    ws.freeze_panes = "D1"


# ---------------------------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------------------------

def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def copy_runs(dst):
    """Light copy of every seed-1 harness run: run-level files + small per-episode logs."""
    n = 0
    for fam in ("luban32", "stack77", "hard45"):
        for name, harness, short, key in AGENTS:
            for v in VARIANTS:
                d = run_dir(fam, key, v)
                src = os.path.join(CC, d)
                out = os.path.join(dst, os.path.basename(d))
                for pat in RUN_KEEP:
                    for f in glob.glob(os.path.join(src, pat)):
                        cp(f, os.path.join(out, os.path.basename(f)))
                for ep in sorted(glob.glob(os.path.join(src, "private", "*"))):
                    for f in os.listdir(ep):
                        if f in EP_KEEP:
                            cp(os.path.join(ep, f), os.path.join(out, "episodes", os.path.basename(ep),
                                                                 "state.json" if f == ".state.json" else f))
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlsx-only", action="store_true", help="write only the workbook")
    a = ap.parse_args()
    root = os.path.abspath(a.out)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    eps6 = exp6_episodes()
    s6 = exp6_summary(eps6)
    st_eps, s1s = exp1_stacking()
    g_flat, s1g = exp1_games()
    fc = fatal_curves(st_eps)
    c2s, need2 = exp2_stacking()
    c2g = exp2_games()

    wb = Workbook()
    ws = wb.active
    ws.title = "Exp1_fatal_steps"
    sheet_exp1(ws, s1s, s1g, fc)
    sheet_exp2(wb.create_sheet("Exp2_budget_search"), c2s, need2, c2g)
    sheet_exp6(wb.create_sheet("Exp6_harness"), s6, eps6)
    xlsx = os.path.join(root, "CHAIN_harness_exp1_2_6.xlsx")
    wb.calculation.fullCalcOnLoad = True          # formulas have no cached values until Excel computes them
    wb.save(xlsx)
    report(s6, xlsx)
    if a.xlsx_only:
        return

    A = HERE
    # exp6
    e6 = os.path.join(root, "exp6_harness")
    write_csv(os.path.join(e6, "exp6_episodes.csv") if os.makedirs(e6, exist_ok=True) is None else "", eps6)
    n = copy_runs(os.path.join(e6, "runs"))
    cp(os.path.join(CC, "hard45_levels.txt"), os.path.join(e6, "levels", "hard45_levels.txt"))
    cp(os.path.join(HERE, "cc_harness", "stacking", "levels.json"), os.path.join(e6, "levels", "stack77_levels.json"))
    for f in ("cc_harness/prompts.py", "cc_harness/stacking/stack_prompt.py", "cc_harness/run_episode.sh",
              "cc_harness/run_episode_codex.sh", "cc_harness/stacking/run_stack_episode.sh",
              "cc_harness/stacking/run_stack_episode_codex.sh", "export_harness_bundle.py"):
        cp(os.path.join(A, f), os.path.join(e6, "code", os.path.basename(f)))
    # exp1
    e1 = os.path.join(root, "exp1_fatal_steps")
    write_csv(os.path.join(e1, "stacking", "exp1_stacking_episodes.csv") if os.makedirs(os.path.join(e1, "stacking"), exist_ok=True) is None else "",
              [{k: v for k, v in e.items() if k != "stretches"} for e in st_eps])
    cp(os.path.join(A, "stacking_oracle/results/exp1_cc/episodes.jsonl"), os.path.join(e1, "stacking", "episodes.jsonl"))
    cp(os.path.join(A, "stacking_oracle/results/exp1_cc/REPORT.md"), os.path.join(e1, "stacking", "REPORT.md"))
    cp(os.path.join(A, "stacking_oracle/results/traj_cc.jsonl"), os.path.join(e1, "stacking", "traj_cc.jsonl"))
    os.makedirs(os.path.join(e1, "three_games"), exist_ok=True)
    write_csv(os.path.join(e1, "three_games", "exp1_three_games_episodes.csv"), g_flat)
    for g, gname in GAMES:
        for f in ("steps.jsonl", "REPORT.md", "summary.json"):
            cp(os.path.join(A, "exp1_vs_agents", g, f), os.path.join(e1, "three_games", gname, f))
        cp(os.path.join(A, "exp1_vs_agents", f"traj_{g}.jsonl"), os.path.join(e1, "three_games", gname, "traj.jsonl"))
    for f in ("fatal_curves.pdf", "fatal_curves.png"):
        if os.path.exists(os.path.join(PAPER, "Fig", f)):
            cp(os.path.join(PAPER, "Fig", f), os.path.join(e1, "figure", f))
    for f in ("stacking_oracle/core.py", "stacking_oracle/collect.py", "stacking_oracle/exp1_stacking.py",
              "envs.py", "exp1_step_value.py", "exp1_vs_agents/collect.py", "plot_fatal_curves.py"):
        cp(os.path.join(A, f), os.path.join(e1, "code", f.replace("/", "__")))
    # exp2
    e2 = os.path.join(root, "exp2_budget_search")
    cp(os.path.join(A, "stacking_oracle/results/runs.jsonl"), os.path.join(e2, "stacking", "search_runs.jsonl"))
    cp(os.path.join(A, "stacking_oracle/results/exp2_REPORT.md"), os.path.join(e2, "stacking", "REPORT.md"))
    with open(os.path.join(e2, "stacking", "success_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["levels", "type", "method_or_agent"] + [f"le_{b}_actions" for b in S_BUDGETS])
        w.writerows(c2s)
    with open(os.path.join(e2, "stacking", "actions_needed_to_match.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "prompt", "agent_success_pct"] + [m for m, _ in S_METHODS])
        w.writerows(need2)
    for g, gname in GAMES:
        for f in ("runs.jsonl", "runs_hard_v1000.jsonl", "REPORT.md"):
            cp(os.path.join(SS, g, "analysis_exp2", f), os.path.join(e2, "three_games", gname, f))
    cp(os.path.join(A, "stacking_oracle/results/three_games_exp2_compare.txt"),
       os.path.join(e2, "three_games", "agents_vs_search.md"))
    with open(os.path.join(e2, "three_games", "success_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "type", "method_or_agent"] + [f"le_{b}_verifications" for b in G_BUDGETS])
        w.writerows(c2g)
    for f in ("stacking_oracle/exp2_stacking.py", "exp2_budget_search.py", "exp2_vs_agents/compare.py"):
        cp(os.path.join(A, f), os.path.join(e2, "code", f.replace("/", "__")))
    print(f"episodes copied: {n}")


def report(s6, xlsx):
    print(f"workbook -> {xlsx}")
    for s in s6:
        if s["prompt"] == "base":
            rates = [100 * w / n_ for w, n_ in s["fam"].values()]
            print(s["agent"], "Pass@1 %.1f" % (sum(rates) / 5), "Succ", sum(w for w, _ in s["fam"].values()),
                  "Steps %.1f NDist %.2f Solv/Tok %.2f" % (s["steps"], s["ndist"],
                                                           sum(w for w, _ in s["fam"].values()) / s["tokens_m"]))


if __name__ == "__main__":
    main()
