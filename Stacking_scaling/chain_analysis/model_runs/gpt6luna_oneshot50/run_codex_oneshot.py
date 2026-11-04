"""Run Codex one-shot on every session in <out>/map.txt: images in, JSON plan out, play_cli plan.

    python run_codex_oneshot.py <out dir with map.txt and prompts/> <model> [workers]

map.txt lines: "<session dir> <level dir>".  Codex runs read-only in an empty
folder with the start images attached; its last line must be the JSON plan.
"""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

O, MODEL = sys.argv[1], sys.argv[2]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPO = os.getcwd()          # run from Stacking_scaling/
ENV = dict(os.environ)      # set HTTPS_PROXY in the environment if Codex needs a proxy
EFFORT = os.environ.get("EFFORT", "high")   # Codex reasoning effort


def last_plan(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("`")
        if line.startswith("[["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return None


def run(line):
    sess, level = line.split()
    num = os.path.basename(sess).split("_")[0]
    if os.path.exists(os.path.join(sess, "final_front.png")):
        return num, "done before"
    if not os.path.exists(os.path.join(sess, ".state.json")):
        subprocess.run(["python3.11", "-m", "chain_analysis.play_cli", "start", level, sess],
                       cwd=REPO, check=True, capture_output=True)
    imgs = sorted(f for f in os.listdir(sess) if f.startswith("obs_000_"))
    empty = tempfile.mkdtemp(prefix="codex_", dir=O)
    args = ["codex", "exec", "-m", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"',
            "--ignore-user-config", "--skip-git-repo-check", "--ephemeral",
            "-s", "read-only", "-C", empty, "--json",
            "-o", os.path.join(O, "answers", num + ".txt")]
    for f in imgs:
        args += ["-i", os.path.join(sess, f)]
    prompt = open(os.path.join(O, "prompts", num + ".md")).read()   # via stdin: -i takes several files
    for attempt in range(3):
        with open(os.path.join(O, "logs", num + ".jsonl"), "w") as log:
            r = subprocess.run(args, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                               input=prompt.encode(), timeout=3600)
        ans = os.path.join(O, "answers", num + ".txt")
        if r.returncode == 0 and os.path.exists(ans):
            break
    else:
        return num, f"codex failed rc={r.returncode}"
    plan = last_plan(open(ans).read())
    raw = json.dumps(plan if plan is not None else [])
    p = subprocess.run(["python3.11", "-m", "chain_analysis.play_cli", "plan", sess, raw],
                       cwd=REPO, capture_output=True, text=True)
    return num, ("no plan found; " if plan is None else "") + p.stdout.strip()


os.makedirs(os.path.join(O, "answers"), exist_ok=True)
os.makedirs(os.path.join(O, "logs"), exist_ok=True)
lines = [l for l in open(os.path.join(O, "map.txt")) if l.strip()]
with ThreadPoolExecutor(WORKERS) as ex:
    for num, msg in ex.map(run, lines):
        print(num, msg, flush=True)
