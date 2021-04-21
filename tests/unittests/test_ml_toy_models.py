import cflearn
import unittest

import numpy as np


x_numerical = np.array([[1.2], [3.4], [5.6]])
x_categorical = np.array([[1.0], [3.0], [5.0]])
x_mix = np.array([xn + xc for xn, xc in zip(x_numerical, x_categorical)])
y_clf = np.array([[1], [1], [2]])
y_reg = np.array([[1.3], [3.5], [5.7]])


class TestMLToy(unittest.TestCase):
    def test_linear_toy(self) -> None:
        cflearn.ml.make_toy_model(
            "linear",
            is_classification=True,
            data_tuple=(x_mix, y_clf),
        )
        cflearn.ml.make_toy_model(
            "linear",
            is_classification=True,
            data_tuple=(x_numerical, y_clf),
        )
        cflearn.ml.make_toy_model(
            "linear",
            is_classification=True,
            data_tuple=(x_categorical, y_clf),
        )
        cflearn.ml.make_toy_model("linear", data_tuple=(x_mix, y_reg))
        cflearn.ml.make_toy_model("linear", data_tuple=(x_numerical, y_reg))
        cflearn.ml.make_toy_model("linear", data_tuple=(x_categorical, y_reg))

    def test_fcnn_toy(self) -> None:
        cflearn.ml.make_toy_model(
            is_classification=True,
            data_tuple=(x_mix, y_clf),
        )
        cflearn.ml.make_toy_model(
            is_classification=True,
            data_tuple=(x_numerical, y_clf),
        )
        cflearn.ml.make_toy_model(
            is_classification=True,
            data_tuple=(x_categorical, y_clf),
        )
        cflearn.ml.make_toy_model(data_tuple=(x_mix, y_reg))
        cflearn.ml.make_toy_model(data_tuple=(x_numerical, y_reg))
        cflearn.ml.make_toy_model(data_tuple=(x_categorical, y_reg))


if __name__ == "__main__":
    unittest.main()
