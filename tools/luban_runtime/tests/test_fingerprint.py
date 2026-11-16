"""The search/runtime source identity must ignore generated outputs and track inputs."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fingerprint_original_engine import source_fingerprint


class SourceFingerprintTests(unittest.TestCase):
    def test_generated_outputs_do_not_change_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "Engine.cs").write_text("original code")
            initial = source_fingerprint(root)
            (root / "obj").mkdir()
            (root / "obj/Generated.cs").write_text("different for each build target")
            (root / "bin").mkdir()
            (root / "bin/Engine.cs").write_text("not a compile input")
            self.assertEqual(initial, source_fingerprint(root))

    def test_build_inputs_and_relative_paths_change_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "Engine.cs").write_text("original code")
            initial = source_fingerprint(root)["sha256"]
            (root / "Directory.Build.props").write_text("compiler configuration")
            with_props = source_fingerprint(root)["sha256"]
            self.assertNotEqual(initial, with_props)
            (root / "Engine.cs").rename(root / "Other.cs")
            self.assertNotEqual(with_props, source_fingerprint(root)["sha256"])


if __name__ == "__main__":
    unittest.main()
