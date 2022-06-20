import os
import unittest


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, "examples", "multimodal")


class TestExample(unittest.TestCase):
    def test_aligner(self) -> None:
        try:
            import cfcv
        except:
            return
        folder = os.path.join(examples_folder, "aligner")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'clip_vqgan.py')} --ci 1"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
