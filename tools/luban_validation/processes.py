"""Stop only the Unity/virtual-display process group created by this validator."""
import os
import signal
import subprocess
import time


def stop_owned_group(process, grace_seconds=8):
    started = time.monotonic()
    result = {'process_group': process.pid, 'forced_group_kill': False}
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    while time.monotonic()-started < grace_seconds:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        # xvfb-run may have exited while Unity is still initializing. Waiting
        # for the shell alone would leave its child consuming resources.
        try:
            os.killpg(process.pid, signal.SIGKILL)
            result['forced_group_kill'] = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    result.update(leader_exit_code=process.returncode, seconds=round(time.monotonic()-started, 3))
    return result
