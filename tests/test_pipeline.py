# type: ignore

import os
import torch
import cflearn
import unittest

import numpy as np


class TestPipeline(unittest.TestCase):
    @staticmethod
    def _build_pipeline() -> cflearn.Pipeline:
        config = cflearn.DLConfig(
            module_name="fcnn",
            module_config=dict(input_dim=1, output_dim=1),
            loss_name="mae",
        )
        return cflearn.Pipeline.init(config)

    def test_cudnn(self) -> None:
        m = self._build_pipeline()
        m.config.cudnn_benchmark = True
        m.build(cflearn.SetDefaultsBlock())
        self.assertTrue(torch.backends.cudnn.benchmark)
        m.config.cudnn_benchmark = False
        m.build(cflearn.SetDefaultsBlock())
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_is_local_rank_0(self) -> None:
        block = cflearn.SetDefaultsBlock()
        self.assertTrue(block.is_local_rank_0)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "0"
        self.assertTrue(block.is_local_rank_0)
        os.environ["LOCAL_RANK"] = "1"
        self.assertFalse(block.is_local_rank_0)
        os.environ["RANK"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        self.assertTrue(block.is_local_rank_0)
        os.environ.pop("RANK")
        os.environ.pop("WORLD_SIZE")
        os.environ.pop("LOCAL_RANK")

    def test_mlflow_auto_callback(self) -> None:
        m = self._build_pipeline()
        m.config.callback_names = ["mlflow"]
        m.build(cflearn.BuildModelBlock(), cflearn.SetTrainerDefaultsBlock())
        mlflow_config = m.config.callback_configs["mlflow"]
        self.assertEqual(mlflow_config["experiment_name"], "fcnn")

    def test_serialization(self) -> None:
        # ml
        x1 = np.random.random([100, 2])
        x2 = np.random.randint(0, 3, [100, 1])
        x = np.hstack([x1, x2, x1])
        y = np.random.random([100, 3])
        m = cflearn.api.fit_ml(
            x,
            y,
            config=cflearn.MLConfig(
                module_name="fcnn",
                module_config=dict(input_dim=x.shape[1], output_dim=y.shape[1]),
                loss_name="mae",
                encoder_settings={"2": cflearn.MLEncoderSettings(3)},
            ),
            debug=True,
        )
        loader = m.data.build_loader(x)
        p = m.predict(loader)[cflearn.PREDICTIONS_KEY]
        name = "test_serialization.ml"
        cflearn.DLPipelineSerializer.save(m, name, compress=True)
        m2 = cflearn.api.load_inference(name)
        loader = m2.data.build_loader(x)
        p2 = m2.predict(loader)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(p, p2)
        cflearn.DLPipelineSerializer.save(m2, name, compress=True)
        m3 = cflearn.api.load_inference(name)
        loader = m3.data.build_loader(x)
        p3 = m3.predict(loader)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(p2, p3)

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
        data = cflearn.MLData.init().fit(x, y)
        for i in range(num_multiple):
            m = cflearn.api.fit_ml(
                data,
                config=cflearn.MLConfig(
                    module_name="fcnn",
                    module_config=dict(
                        input_dim=data.num_features,
                        output_dim=data.num_labels,
                    ),
                    loss_name="mae",
                    encoder_settings={
                        "0": cflearn.MLEncoderSettings(3),
                        "3": cflearn.MLEncoderSettings(11),
                    },
                ),
                debug=True,
            )
            loader = data.build_loader(x)
            ps.append(m.predict(loader)[cflearn.PREDICTIONS_KEY])
            packed = f"./m{i}"
            cflearn.DLPipelineSerializer.save(m, packed, compress=True)
            packed_list.append(packed)
        fused = cflearn.api.fuse_inference(packed_list)
        loader = fused.data.build_loader(x)
        pf = fused.predict(loader)[cflearn.PREDICTIONS_KEY]
        assert np.allclose(np.stack(ps).mean(axis=0), pf)


if __name__ == "__main__":
    unittest.main()
