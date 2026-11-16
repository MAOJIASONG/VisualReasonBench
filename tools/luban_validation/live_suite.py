"""Real Unity load/render/action checks; never restores snapshots or sets victory flags."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback

from PIL import Image, ImageStat
from processes import stop_owned_group


class Client:
    def __init__(self, out: Path, port=9999, timeout=130):
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.buffer = b""
        self.log = (out / "commands.jsonl").open("a", buffering=1)

    def close(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        self.buffer = b""

    def call(self, command, params=None, env=1):
        request = {"command": command, "env_id": env, "params": params or {}}
        start = time.monotonic()
        receipt = {"request": request, "started_unix": time.time()}
        try:
            if self.sock is None:
                self.sock = socket.create_connection(("127.0.0.1", self.port), self.timeout)
            self.sock.sendall((json.dumps(request) + "\n").encode())
            while b"\n" not in self.buffer:
                part = self.sock.recv(1048576)
                if not part:
                    raise ConnectionError("Server closed before complete JSON reply")
                self.buffer += part
            line, self.buffer = self.buffer.split(b"\n", 1)
            if not line.strip():
                raise ValueError("Empty server response (not a successful action)")
            response = json.loads(line)
            receipt["response"] = response
            return response
        except BaseException as exc:
            receipt["error"] = repr(exc)
            # Never retry an action: it might have executed before the reply failed.
            self.close()
            raise
        finally:
            receipt["seconds"] = round(time.monotonic() - start, 5)
            self.log.write(json.dumps(receipt, ensure_ascii=False) + "\n")

    def success(self, command, params=None):
        response = self.call(command, params)
        if response.get("status") != "success":
            raise RuntimeError(f"{command}: {response}")
        return response.get("data", {})


def copy_captures(data, package, out, level):
    paths = [p for p in data.get("screenshot_path", "").split("|") if p]
    if len(paths) != 6:
        raise AssertionError(f"Expected 6 viewpoints, got {len(paths)}")
    images = []
    for index, name in enumerate(paths):
        source = Path(name).resolve()
        if not source.is_relative_to(package):
            raise ValueError(f"Screenshot outside runtime package: {source}")
        destination = out / "screenshots" / f"level_{level:02d}_view_{index}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        with Image.open(destination) as im:
            im.load()
            deviation = ImageStat.Stat(im.convert("RGB")).stddev
            if max(deviation) <= 1:
                raise AssertionError(f"Blank screenshot: {destination}")
            images.append({"file": str(destination), "size": list(im.size), "stddev": deviation,
                           "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    return images


def run(args):
    package = args.package.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "package": str(package), "levels": [], "mode": "native",
              "assembly_sha256": hashlib.sha256((package / "LuBanSuoEnv_Data/Managed/Assembly-CSharp.dll").read_bytes()).hexdigest()}
    def save():
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    manifest_path = package / 'repair_manifest.json'
    timeout = json.loads(manifest_path.read_text())['response_timeout_ms']/1000+10 if manifest_path.exists() else 130
    client = Client(out, args.port, timeout=timeout)
    proc = None
    try:
        try:
            with socket.create_connection(("127.0.0.1", args.port), 1):
                raise RuntimeError("An existing server is listening; refusing to touch it")
        except ConnectionRefusedError:
            pass
        command = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", str(package / "LuBanSuoEnv.x86_64"),
                   "-job-worker-count", "2", "-logFile", str(out / "unity.log")]
        report["launch_command"] = command
        proc = subprocess.Popen(command, cwd=package, stdout=(out / "launcher.log").open("w"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        report["pid"] = proc.pid
        started = time.monotonic()
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError(f"Unity launcher exited early with {proc.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", args.port), .5):
                    break
            except OSError:
                time.sleep(.5)
        else:
            raise TimeoutError("Unity did not start within 90 seconds")
        report["startup_seconds"] = round(time.monotonic() - started, 3)
        print("UNITY READY", report["startup_seconds"], flush=True)
        for level in args.levels:
            row = {"level": level, "load_ok": False, "capture_ok": False}
            report["levels"].append(row)
            try:
                client.success("start_level", {"level_index": level})
                state = client.success("get_state")
                expected = json.loads((args.level_data / f"Puzzle_{level}.json").read_text())
                if state.get("puzzle_index") != level or len(state.get("pieces", [])) != len(expected["polyhedrons"]):
                    raise AssertionError(f"Loaded wrong level or piece count: {state}")
                row.update(load_ok=True, initial_state=state)
                row["screenshots"] = copy_captures(client.success("capture"), package, out, level)
                row["capture_ok"] = True
                candidate_path = args.candidates / f"Puzzle_{level}.json" if args.candidates else None
                if candidate_path and candidate_path.exists():
                    candidate = json.loads(candidate_path.read_text())
                    if candidate.get("status") == "solved":
                        row["candidate_source"] = str(candidate_path)
                        row["replay"] = []
                        for move in candidate["moves"]:
                            result = client.call("move", {"piece_id": move["piece_id"], "axis": move["axis"],
                                                          "distance": move["distance"], "is_exploratory": False})
                            if result.get("status") != "success":
                                raise AssertionError(f"Replay move rejected: {result}")
                            time.sleep(.6)
                            row["replay"].append({"move": move, "state": client.success("get_state")})
                        final = client.success("get_state")
                        row["native_is_solved"] = final.get("is_solved") is True
                        if not row["native_is_solved"]:
                            raise AssertionError("Candidate did not trigger native game victory")
                print("LEVEL", level, "load/capture OK", "native_solved", row.get("native_is_solved"), flush=True)
            except Exception as exc:
                row["error"] = repr(exc)
                print("LEVEL", level, "ERROR", repr(exc), flush=True)
            save()
        if args.protocol_tests:
            tests = report["protocol_regressions"] = {}
            try:
                client.success("start_level", {"level_index": 0})
                client.success("move", {"piece_id": 0, "axis": "z", "distance": -300})
                time.sleep(.6)
                before = client.success("get_state")
                client.success("rotate", {"piece_id": 0, "axis": "x", "angle": 180})
                time.sleep(.6)
                after = client.success("get_state")
                tests["rotation_180"] = {
                    "before": before, "after": after,
                    "passed": after["pieces"][0]["rot"] == "XPositive_YPositive_ZPositive",
                }
                rejected = client.call("rotate", {"piece_id": 0, "axis": "x", "angle": 45})
                unchanged = client.success("get_state")
                tests["native_45_rejected"] = {
                    "response": rejected,
                    "passed": rejected.get("status") != "success" and
                              unchanged["pieces"][0]["rot"] == after["pieces"][0]["rot"],
                }
            except Exception as exc:
                tests["error"] = repr(exc)
        report["complete"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        traceback.print_exc()
    finally:
        client.close()
        if proc is not None:
            report["process_cleanup"] = stop_owned_group(proc)
            report["launcher_exit_code"] = proc.returncode
        try:
            with socket.create_connection(("127.0.0.1", args.port), .5):
                report["port_closed_after_test"] = False
        except OSError:
            report["port_closed_after_test"] = True
        save()
    passed = report["complete"] and all(not row.get("error") for row in report["levels"])
    if args.protocol_tests:
        tests = report.get("protocol_regressions", {})
        passed = passed and all(tests.get(key, {}).get("passed") for key in ["rotation_180", "native_45_rejected"])
    print("FINAL", "PASS" if passed else "FAIL", "levels", len(report["levels"]), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level-data", type=Path, default=Path(__file__).resolve().parents[1] / "luban_solver/levels")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--protocol-tests", action="store_true")
    raise SystemExit(run(parser.parse_args()))
