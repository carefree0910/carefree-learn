import cflearn

import numpy as np

from cfdata.tabular import TabularDataset


def test_onnx():
    def _core(dataset):
        x, y = dataset.xy
        m = cflearn.make(model, verbose_level=0, use_tqdm=False).fit(x, y)
        predictions = m.predict(x)
        cflearn.ONNX(m).to_onnx("m.onnx").inject_onnx()
        assert np.allclose(predictions, m.predict(x), atol=1e-4, rtol=1e-4)

    reg_models = ["linear", "fcnn", "tree_dnn"]
    for model in reg_models:
        _core(TabularDataset.boston())
    clf_models = reg_models + ["nnb", "ndt"]
    for model in clf_models:
        _core(TabularDataset.iris())


if __name__ == "__main__":
    test_onnx()
