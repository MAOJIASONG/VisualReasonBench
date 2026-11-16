import os
import select
import signal
import subprocess
import sys
import unittest
from processes import stop_owned_group


class OwnedProcessCleanup(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'Linux Unity validator')
    def test_child_is_stopped_after_launcher_exits(self):
        child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(30)"
        parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(30)"
        process = subprocess.Popen([sys.executable, '-c', parent], start_new_session=True,
                                   stdout=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0], 'Child did not start')
            self.assertEqual(process.stdout.readline().strip(), 'ready')
            result = stop_owned_group(process, grace_seconds=.1)
            self.assertTrue(result['forced_group_kill'])
            # The child inherited the pipe: EOF proves it also exited.
            self.assertEqual(process.stdout.read(), '')
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
