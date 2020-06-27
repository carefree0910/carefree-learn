import cflearn
import unittest

import numpy as np

from cfdata.tabular import *
from sklearn.naive_bayes import *
from cftool.ml import ModelPattern
from sklearn.tree import DecisionTreeClassifier


class TestTraditional(unittest.TestCase):
    @staticmethod
    def _train_traditional(model, dataset, sklearn_model):
        m = cflearn.make(model, cv_split=0., num_epoch=1, max_epoch=2)
        m0 = cflearn.make(model, cv_split=0., num_epoch=0, max_epoch=0)
        m.fit(*dataset.xy)
        m0.fit(*dataset.xy)
        metrics = ["acc", "auc"]
        cflearn.estimate(*dataset.xy, metrics=metrics, wrappers={"fit": m, "init": m0})
        x, y = m0.tr_data.processed.xy
        split = m0.model.get_split(x, m0.device)
        x, sk_y = split.merge().cpu().numpy(), y.ravel()
        sklearn_model.fit(x, sk_y)
        pattern = ModelPattern(
            init_method=lambda: sklearn_model,
            predict_method=lambda x_: sklearn_model.predict(x_).reshape([-1, 1]),
            predict_prob_method="predict_proba"
        )
        cflearn.estimate(x, y, metrics=metrics, other_patterns={"sklearn": pattern})
        return m, m0, x

    def test_nnb_gnb(self):
        gnb = GaussianNB()
        dataset = TabularDataset.iris()
        nnb, nnb0, x = self._train_traditional("nnb", dataset, gnb)
        self.assertTrue(np.allclose(nnb0.model.class_prior, gnb.class_prior_))
        self.assertTrue(np.allclose(nnb0.model.mu.data.cpu().numpy(), gnb.theta_))
        self.assertTrue(np.allclose(nnb0.model.std.data.cpu().numpy() ** 2, gnb.sigma_))
        self.assertTrue(np.allclose(nnb0.predict_prob(dataset.x), gnb.predict_proba(x)))

    def test_nnb_mnb(self):
        mnb = MultinomialNB()
        dataset = TabularDataset.digits()
        nnb, nnb0, x = self._train_traditional("nnb", dataset, mnb)
        self.assertTrue(np.allclose(nnb0.model.class_log_prior(numpy=True), mnb.class_log_prior_))
        self.assertTrue(np.allclose(nnb0.predict_prob(dataset.x), mnb.predict_proba(x), atol=1e-4))

    def test_ndt(self):
        dt = DecisionTreeClassifier()
        self._train_traditional("ndt", TabularDataset.iris(), dt)
        self._train_traditional("ndt", TabularDataset.digits(), dt)
        self._train_traditional("ndt", TabularDataset.breast_cancer(), dt)


if __name__ == '__main__':
    unittest.main()
