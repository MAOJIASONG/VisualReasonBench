"""Gather the 10 agent runs on the 45 hard levels into one Exp-1 trajectory file per game
(last try per level when a level was rerun):  python chain_analysis/exp1_vs_agents/collect.py"""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "cc_runs"))
RUNS = [("hard45_sonnet_base", "Sonnet base"), ("hard45_sonnet_plan", "Sonnet plan"),
        ("hard45_opus_base", "Opus base"), ("hard45_opus_plan", "Opus plan"),
        ("hard45_codex_base_gpt-6.1-sol", "sol base"), ("hard45_codex_plan_gpt-6.1-sol", "sol plan"),
        ("hard45_codex_base_gpt-6.1-terra", "terra base"), ("hard45_codex_plan_gpt-6.1-terra", "terra plan"),
        ("hard45_codex_base_gpt-6-luna", "luna base"), ("hard45_codex_plan_gpt-6-luna", "luna plan")]
for game in ("type_3", "type_3_1", "type_2_3"):
    n = 0
    with open(os.path.join(HERE, f"traj_{game}.jsonl"), "w") as f:
        for d, name in RUNS:
            last = {}
            for l in open(f"{CC}/{d}/traj_{game}.jsonl"):
                if l.strip():
                    t = json.loads(l)
                    last[os.path.basename(os.path.normpath(t["level"]))] = t
            for lv, t in sorted(last.items()):
                f.write(json.dumps({"model": name, "level": lv, "actions": t["actions"]}) + "\n")
                n += 1
    print(game, n)
