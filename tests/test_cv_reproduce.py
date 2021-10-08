import os
import unittest


file_folder = os.path.dirname(__file__)
scripts_folder = os.path.join(file_folder, os.pardir, "examples", "cv", "reproduce")


class TestReproduce(unittest.TestCase):
    def _core(self, sub_folder: str) -> None:
        folder = os.path.join(scripts_folder, sub_folder)
        self.assertEqual(
            os.system(f"python {os.path.join(folder, f'run_{sub_folder}.py')}"),
            0,
        )

    def test_u2net(self) -> None:
        self._core("u2net")


if __name__ == "__main__":
    unittest.main()
