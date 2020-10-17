import cflearn

import numpy as np

from cfdata.tabular import TabularDataset


def test_onnx() -> None:
    def _core(dataset: TabularDataset) -> None:
        x, y = dataset.xy
        m = cflearn.make(
            model,
            verbose_level=0,
            use_tqdm=False,
            min_epoch=1,
            num_epoch=2,
            max_epoch=4,
        )
        m.fit(x, y)
        predictions = m.predict(x)
        predictor_folder = "test_onnx"
        cflearn.Pack.pack(m, predictor_folder)
        predictor = cflearn.Pack.get_predictor(predictor_folder)
        assert np.allclose(predictions, predictor.predict(x), atol=1e-4, rtol=1e-4)

    reg_models = ["linear", "fcnn", "tree_dnn"]
    for model in reg_models:
        _core(TabularDataset.boston())
    clf_models = reg_models + ["nnb", "ndt"]
    for model in clf_models:
        _core(TabularDataset.iris())


if __name__ == "__main__":
    test_onnx()
