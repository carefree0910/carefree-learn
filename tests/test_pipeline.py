import os
import json
import torch
import cflearn
import unittest

import numpy as np

from cftool.misc import get_latest_workplace


class TestPipeline(unittest.TestCase):
    def test_cudnn(self) -> None:
        cflearn.api.resnet18_gray(2, cudnn_benchmark=True)
        self.assertTrue(torch.backends.cudnn.benchmark)

    def test_is_rank_0(self) -> None:
        m = cflearn.api.resnet18_gray(2)
        self.assertTrue(m.is_rank_0)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "0"
        self.assertTrue(m.is_rank_0)
        os.environ["LOCAL_RANK"] = "1"
        self.assertTrue(m.is_rank_0)
        os.environ["RANK"] = "1"
        self.assertFalse(m.is_rank_0)
        os.environ.pop("RANK")
        os.environ.pop("WORLD_SIZE")
        os.environ.pop("LOCAL_RANK")

    def test_mlflow_auto_callback(self) -> None:
        m = cflearn.api.resnet18_gray(2, build=True, callback_names="mlflow")
        callback_configs = m.trainer_config.callback_configs
        assert callback_configs is not None
        mlflow_config = callback_configs["mlflow"]
        self.assertEqual(mlflow_config["experiment_name"], "clf")

    def test_serialization(self) -> None:
        # ml
        x1 = np.random.random([100, 2])
        x2 = np.random.randint(0, 3, [100, 1])
        x = np.hstack([x1, x2, x1])
        y = np.random.random([100, 3])
        m = cflearn.api.fit_ml(
            x,
            y,
            is_classification=False,
            config=cflearn.MLConfig(output_dim=3, encoding_settings={2: dict(dim=3)}),
            debug=True,
        )
        idata = m.make_inference_data(x)
        p = m.predict(idata)[cflearn.PREDICTIONS_KEY]
        name = "test_serialization.ml"
        m.save(name)
        m2: cflearn.ml.MLPipeline = cflearn.api.load(name)
        idata = m2.make_inference_data(x)
        p2 = m2.predict(idata)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(p, p2)
        m2.save(name)
        m3: cflearn.ml.MLPipeline = cflearn.api.load(name)
        idata = m3.make_inference_data(x)
        p3 = m3.predict(idata)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(p2, p3)
        # cv
        n = 3
        size = 7
        m = cflearn.api.cct_lite(size, 10, metric_names="acc", debug=True)
        x = torch.randn(n, 3, size, size)
        y = torch.randint(0, 2, [n, 1])
        data = cflearn.TensorData(x, y)
        name = "test_serialization.cv"
        m.fit(data).save(name)
        m2 = cflearn.api.load(name)
        p = m.predict(data)[cflearn.PREDICTIONS_KEY]
        p2 = m2.predict(data)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(p, p2)
        m2.save(name)
        m3 = cflearn.api.load(name)
        p3 = m3.predict(data)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(p2, p3)
        workplace = get_latest_workplace("_logs")
        assert workplace is not None
        cflearn.api.pack_onnx(
            workplace,
            ".onnx",
            input_sample={cflearn.INPUT_KEY: torch.randn(1, 3, size, size)},
        )

    def test_model_soup(self) -> None:
        portion = 0.01
        m = cflearn.api.cct_lite(
            28,
            10,
            valid_portion=portion,
            fixed_steps=10,
            log_steps=1,
        )
        data = cflearn.MNISTData(batch_size=4, transform=["to_rgb", "to_tensor"])
        m.fit(data)
        valid_loader = m.trainer.valid_loader
        workplace = get_latest_workplace("_logs")
        packed = cflearn.api.pack(
            workplace,
            compress=False,
            model_soup_loader=valid_loader,
            model_soup_metric_names=["acc", "auc"],
            model_soup_valid_portion=portion,
        )
        m = cflearn.api.load(packed, compress=False)
        res = m.inference.get_outputs(
            valid_loader,  # type: ignore
            portion=portion,
            metrics=cflearn._IMetric.fuse(["acc", "auc"]),
        )
        with open(os.path.join(packed, cflearn.SCORES_FILE), "r") as f:
            score = list(json.load(f).values())[0]
        self.assertAlmostEqual(res.metric_outputs.final_score, score)  # type: ignore

    def test_fuse_multiple(self) -> None:
        n = 100
        num_multiple = 7
        x = np.hstack(
            [
                np.random.randint(0, 3, [n, 1]),
                np.random.random([n, 2]),
                np.random.randint(0, 11, [n, 1]),
                np.random.random([n, 3]),
            ]
        )
        y = np.random.random([n, 5])
        ps = []
        packed_list = []
        for i in range(num_multiple):
            m = cflearn.api.fit_ml(
                x,
                y,
                is_classification=False,
                config=cflearn.MLConfig(
                    output_dim=5,
                    encoding_settings={0: dict(dim=3), 3: dict(dim=11)},
                ),
                debug=True,
            )
            idata = m.make_inference_data(x)
            ps.append(m.predict(idata)[cflearn.PREDICTIONS_KEY])
            packed = f"./m{i}"
            m.save(packed)
            packed_list.append(packed)
        fused = cflearn.ml.fuse_multiple(packed_list)
        idata = fused.make_inference_data(x)
        pf = fused.predict(idata)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(np.stack(ps).mean(axis=0), pf)


if __name__ == "__main__":
    unittest.main()
