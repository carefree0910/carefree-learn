import os
import unittest


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, "examples", "cv")


class TestExample(unittest.TestCase):
    def test_mnist_clf(self) -> None:
        folder = os.path.join(examples_folder, "classification")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_clf.py')} --ci 1"),
            0,
        )

    def test_mnist_gan(self) -> None:
        folder = os.path.join(examples_folder, "gan")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_gan.py')} --ci 1"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
