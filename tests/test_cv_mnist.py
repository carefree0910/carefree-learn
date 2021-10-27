import os
import unittest


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, "examples", "cv", "mnist")


class TestMNIST(unittest.TestCase):
    def test_clf(self) -> None:
        self.assertEqual(
            os.system(f"python {os.path.join(examples_folder, 'run_clf.py')} --ci 1"),
            0,
        )

    def test_vae(self) -> None:
        self.assertEqual(
            os.system(f"python {os.path.join(examples_folder, 'run_vae.py')} --ci 1"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
