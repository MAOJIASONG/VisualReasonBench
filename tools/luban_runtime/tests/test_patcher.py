"""Run with DOTNET=/path/to/dotnet python -m unittest discover -s tools/luban_runtime/tests.

These test safe package construction, not gameplay. Live action/capture validation is
separate and must use the native player and its actual completion receipts.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
DOTNET = os.environ.get("DOTNET", "dotnet")
PATCHER = ROOT / "tools/luban_runtime/patcher/bin/Release/net8.0/RuntimePatcher.dll"
SOURCE = ROOT / "Luban/LuBanSuoEnv_Data/Managed/Assembly-CSharp.dll"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PatcherSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="luban-patcher-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def run_patch(self, source, output, timeout=None):
        command = [DOTNET, str(PATCHER), str(source), str(output)]
        if timeout is not None:
            command.append(str(timeout))
        return subprocess.run(command,
                              capture_output=True, text=True)

    def test_rejects_in_place_without_changing_input(self):
        before = digest(SOURCE)
        result = self.run_patch(SOURCE, SOURCE)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(digest(SOURCE), before)

    def test_rejects_unknown_input_before_writing_output(self):
        source, output = self.folder / "unknown.dll", self.folder / "output.dll"
        source.write_bytes(SOURCE.read_bytes() + b"unsupported change")
        result = self.run_patch(source, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported input assembly hash", result.stderr)
        self.assertFalse(output.exists())

    def test_rejects_existing_output(self):
        output = self.folder / "existing.dll"
        output.write_bytes(b"keep this file")
        result = self.run_patch(SOURCE, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), b"keep this file")

    def test_reproducible_output_and_source_unchanged(self):
        before = digest(SOURCE)
        outputs = [self.folder / "first.dll", self.folder / "second.dll"]
        receipts = []
        for output in outputs:
            result = self.run_patch(SOURCE, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["response_timeout_ms"], 300000)
            self.assertEqual(receipt["output_sha256"], digest(output))
            receipts.append(receipt)
        self.assertEqual(digest(outputs[0]), digest(outputs[1]))
        self.assertEqual(before, digest(SOURCE))
        self.assertFalse(receipts[0]["original_mode_extension"])

    def test_rejects_unsupported_timeout(self):
        for timeout in [0, 600001]:
            output = self.folder / f"timeout_{timeout}.dll"
            result = self.run_patch(SOURCE, output, timeout)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
