from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform == "win32", "Windows launcher test")
class WindowsLauncherTests(unittest.TestCase):
    def test_parentheses_in_install_path_do_not_break_batch_parsing(self):
        source = Path(__file__).resolve().parents[1] / "Start Marginalia.cmd"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Marginalia download (2)"
            root.mkdir()
            shutil.copy2(source, root / source.name)
            (root / "requirements.txt").write_text("", encoding="utf-8")
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            (fake_bin / "py.cmd").write_text("@exit /b 0\r\n", encoding="ascii")
            environment = os.environ.copy()
            environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "call", str(root / source.name)],
                cwd=root,
                env=environment,
                input="\n",
                text=True,
                capture_output=True,
                timeout=15,
            )
        output = result.stdout + result.stderr
        self.assertNotIn("was unexpected at this time", output)
        self.assertIn(str(root / ".venv"), output)


if __name__ == "__main__":
    unittest.main()
