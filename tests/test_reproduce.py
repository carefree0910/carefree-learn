import os
import unittest

from typing import Optional


file_folder = os.path.dirname(__file__)
scripts_folder = os.path.join(file_folder, os.pardir, "examples", "reproduce")


class TestReproduce(unittest.TestCase):
    def _core(self, sub_folder: str, name: Optional[str] = None) -> None:
        folder = os.path.join(scripts_folder, sub_folder)
        file = f"run_{name or sub_folder}.py"
        self.assertEqual(os.system(f"python {os.path.join(folder, file)} --ci 1"), 0)

    def test_clip(self) -> None:
        try:
            import PIL
            import requests
            import transformers
        except:
            return
        self._core("clip", name="base")
        # self._core("clip", name="chinese")
        # self._core("clip", name="open_clip")


if __name__ == "__main__":
    unittest.main()
