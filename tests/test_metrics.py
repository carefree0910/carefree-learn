import cflearn
import unittest

import numpy as np

from cflearn.schema import _IMetric
from cflearn.constants import LABEL_KEY
from cflearn.constants import PREDICTIONS_KEY


def _get_score(
    metric: _IMetric,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    return metric.evaluate(
        {LABEL_KEY: labels},
        {PREDICTIONS_KEY: predictions},
    ).final_score


class TestMetrics(unittest.TestCase):
    def test_acc(self) -> None:
        acc = cflearn.api.make_metric("acc")
        labels = np.array([1, 1, 1, 1]).reshape([4, 1])
        p1 = np.array([-0.2, -0.1, 0.1, 0.2]).reshape([4, 1])
        p2 = np.array([[-0.2, -0.1, 0.1, 0.2], [0.0, 0.0, 0.0, 0.0]]).reshape([4, 2])
        p3 = np.array(
            [
                [-0.2, -0.1, 0.2],
                [-0.1, 0.1, 0.2],
                [-0.2, 0.2, 0.1],
                [0.1, 0.2, -0.1],
            ]
        )
        self.assertAlmostEqual(_get_score(acc, p1, labels), 0.5)
        self.assertAlmostEqual(_get_score(acc, p2, labels), 0.5)
        self.assertAlmostEqual(_get_score(acc, p3, labels), 0.5)

    def test_quantile(self) -> None:
        q1 = cflearn.api.make_metric("quantile", q=0.1)
        q3 = cflearn.api.make_metric("quantile", q=0.3)
        q5 = cflearn.api.make_metric("quantile", q=0.5)
        q7 = cflearn.api.make_metric("quantile", q=0.7)
        q9 = cflearn.api.make_metric("quantile", q=0.9)
        labels = np.linspace(0, 1, 11).reshape([11, 1])
        p1 = np.full_like(labels, 0.1)
        p3 = np.full_like(labels, 0.3)
        p5 = np.full_like(labels, 0.5)
        p7 = np.full_like(labels, 0.7)
        p9 = np.full_like(labels, 0.9)
        self.assertAlmostEqual(_get_score(q1, p1, labels), -0.04909090909090909)
        self.assertAlmostEqual(_get_score(q3, p1, labels), -0.12909090909090912)
        self.assertAlmostEqual(_get_score(q3, p3, labels), -0.11454545454545455)
        self.assertAlmostEqual(_get_score(q3, p5, labels), -0.13636363636363635)
        self.assertAlmostEqual(_get_score(q5, p3, labels), -0.15454545454545457)
        self.assertAlmostEqual(_get_score(q5, p5, labels), -0.13636363636363635)
        self.assertAlmostEqual(_get_score(q5, p7, labels), -0.15454545454545457)
        self.assertAlmostEqual(_get_score(q7, p5, labels), -0.13636363636363635)
        self.assertAlmostEqual(_get_score(q7, p7, labels), -0.11454545454545456)
        self.assertAlmostEqual(_get_score(q7, p9, labels), -0.12909090909090912)
        self.assertAlmostEqual(_get_score(q9, p7, labels), -0.07454545454545457)
        self.assertAlmostEqual(_get_score(q9, p9, labels), -0.04909090909090908)

    def test_mae(self) -> None:
        mae = cflearn.api.make_metric("mae")
        labels = np.array([1, 1, 1, 1]).reshape([4, 1])
        predictions = np.array([-0.2, -0.1, 0.1, 0.2]).reshape([4, 1])
        self.assertAlmostEqual(_get_score(mae, predictions, labels), -1.0)

    def test_mse(self) -> None:
        mse = cflearn.api.make_metric("mse")
        labels = np.array([1, 1, 1, 1]).reshape([4, 1])
        predictions = np.array([-0.2, -0.1, 0.1, 0.2]).reshape([4, 1])
        self.assertAlmostEqual(_get_score(mse, predictions, labels), -1.025)


if __name__ == "__main__":
    unittest.main()
