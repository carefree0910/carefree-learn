import os
import unittest


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, os.pardir, "examples", "ml")


class TestExample(unittest.TestCase):
    def test_iris(self) -> None:
        folder = os.path.join(examples_folder, "iris")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_iris.py')}"),
            0,
        )

    def test_operations(self) -> None:
        folder = os.path.join(examples_folder, "operations")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_op.py')}"),
            0,
        )

    def test_simple(self) -> None:
        folder = os.path.join(examples_folder, "simple")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'boston.py')}"),
            0,
        )
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'iris.py')}"),
            0,
        )

    def test_titanic(self) -> None:
        folder = os.path.join(examples_folder, "titanic")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_titanic.py')}"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
