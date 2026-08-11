from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipIf(sys.platform == "win32", "POSIX launcher test")
class UnixLauncherTests(unittest.TestCase):
    def test_packaged_layout_works_from_path_with_spaces_and_parentheses(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Marginalia download (2)"
            app_root = root / "marginalia"
            python = app_root / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            shutil.copy2(repository / "Start Marginalia.sh", root / "Start Marginalia.sh")
            (app_root / "launcher.py").write_text("", encoding="utf-8")
            (app_root / "requirements.txt").write_text("", encoding="utf-8")
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(python, 0o755)
            result = subprocess.run(
                ["/bin/sh", str(root / "Start Marginalia.sh")],
                cwd=root,
                env={**os.environ, "MARGINALIA_SETUP_ONLY": "1"},
                text=True,
                capture_output=True,
                timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_macos_command_delegates_to_shared_launcher(self):
        command = (Path(__file__).resolve().parents[1] / "Start Marginalia.command").read_text(encoding="utf-8")
        self.assertIn('exec /bin/sh "$SCRIPT_DIR/Start Marginalia.sh"', command)


if __name__ == "__main__":
    unittest.main()
