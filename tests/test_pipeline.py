import os
import torch
import cflearn
import unittest

from cflearn.misc.toolkit import get_latest_workplace


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
        m = cflearn.api.resnet18_gray(2, callback_names="mlflow")
        callback_configs = m.trainer_config["callback_configs"]
        mlflow_config = callback_configs["mlflow"]
        self.assertEqual(mlflow_config["experiment_name"], "clf")

    def test_serialization(self) -> None:
        m = cflearn.api.cct_lite(28, 10, metric_names="acc", debug=True)
        data = cflearn.cv.MNISTData(batch_size=4, transform=["to_rgb", "to_tensor"])
        name = "test_serialization"
        m.fit(data).save(name)
        cflearn.api.load(name)
        workplace = get_latest_workplace("_logs")
        assert workplace is not None
        cflearn.api.pack_onnx(
            workplace,
            ".onnx",
            input_sample={cflearn.INPUT_KEY: torch.randn(1, 3, 28, 28)},
        )


if __name__ == "__main__":
    unittest.main()
