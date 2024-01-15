import os
import unittest


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, "examples", "ml")


class TestExample(unittest.TestCase):
    def test_simple(self) -> None:
        try:
            import sklearn
        except:
            return
        folder = os.path.join(examples_folder, "simple")
        for file in ["iris", "california", "toy"]:
            path = os.path.join(folder, f"{file}.py")
            self.assertEqual(os.system(f"python {path} --ci 1"), 0)

    def test_mlflow(self) -> None:
        try:
            import mlflow
            import sklearn
        except:
            return
        path = os.path.join(examples_folder, "mlflow", "california_with_mlflow.py")
        self.assertEqual(os.system(f"python {path} --ci 1"), 0)
        self.assertTrue(os.path.isdir("mlruns"))

    def test_titanic(self) -> None:
        folder = os.path.join(examples_folder, "titanic")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_titanic.py')}"),
            0,
        )
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_titanic_ddp.py')}"),
            0,
        )

    def test_titanic_interpret(self) -> None:
        try:
            import captum
            import matplotlib
        except:
            return
        folder = os.path.join(examples_folder, "titanic")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'test_titanic_interpret.py')}"),
            0,
        )

    def test_operations(self) -> None:
        folder = os.path.join(examples_folder, "operations")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_op.py')} --ci 1"),
            0,
        )

    def test_iris(self) -> None:
        folder = os.path.join(examples_folder, "iris")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_iris.py')} --ci 1"),
            0,
        )

    def test_ddr(self) -> None:
        folder = os.path.join(examples_folder, "ddr")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'run_ddr.py')} --ci 1"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
