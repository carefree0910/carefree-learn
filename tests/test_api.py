import os
import unittest
import subprocess


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, "examples")


class TestAPI(unittest.TestCase):
    def test_run_multiple(self) -> None:
        launcher_folder = os.path.join(examples_folder, "run_multiple")
        launcher_path = os.path.join(launcher_folder, "launcher.py")
        cmd = ["python", launcher_path, "--ci", "1"]
        result = subprocess.run(cmd, cwd=launcher_folder)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
