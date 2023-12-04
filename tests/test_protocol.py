import torch
import cflearn
import unittest

from typing import Any
from cflearn.schema import ILoss
from cflearn.losses import build_loss
from cflearn.pipeline import MLTrainingPipeline


def _get_m(**kwargs: Any) -> MLTrainingPipeline:
    return cflearn.api.make_toy_ml_model(**kwargs)


class TestProtocol(unittest.TestCase):
    def test_mlflow_callback(self) -> None:
        m = _get_m(callback_names="mlflow")
        callbacks = [c.__identifier__ for c in m.build_trainer.trainer.callbacks]
        self.assertTrue("mlflow" in callbacks)

    def test_disable_logging(self) -> None:
        m = _get_m()
        state = m.build_trainer.trainer.state
        self.assertTrue(state.enable_logging)
        with state.disable_logging:
            self.assertFalse(state.enable_logging)
        self.assertTrue(state.enable_logging)

    def test_loss_protocol(self) -> None:
        def _get_loss(loss_ins: ILoss) -> float:
            return loss_ins.run(forward_results, batch)[cflearn.LOSS_KEY].item()

        predictions = torch.full([10, 1], 2.0)
        labels = torch.zeros([10, 1])
        forward_results = {cflearn.PREDICTIONS_KEY: predictions}
        batch = {cflearn.LABEL_KEY: labels}
        self.assertEqual(_get_loss(build_loss("mae")), 2.0)
        self.assertEqual(_get_loss(build_loss("mae", reduction="sum")), 20)
        mae = cflearn.MAELoss()
        self.assertEqual(mae(predictions, labels).mean().item(), 2.0)
        self.assertEqual(mae(predictions, labels).sum().item(), 20.0)
        # multi task
        multi_task_loss = build_loss("multi_task", loss_names=["mae", "mse"])
        losses = multi_task_loss.run(forward_results, batch)
        self.assertEqual(losses[cflearn.LOSS_KEY].item(), 6.0)
        self.assertEqual(losses["mae"].item(), 2.0)
        self.assertEqual(losses["mse"].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
