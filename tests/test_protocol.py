import torch
import cflearn
import unittest

from typing import Any
from cflearn.pipeline import DLPipeline


def _get_m(**kwargs: Any) -> DLPipeline:
    kw = dict(metric_names="acc", debug=True)
    kw.update(kwargs)
    m = cflearn.api.cct_lite(28, 10, **kw)  # type: ignore
    data = cflearn.cv.MNISTData(batch_size=4, transform=["to_rgb", "to_tensor"])
    return m.fit(data)


class TestProtocol(unittest.TestCase):
    def test_mlflow_callback(self) -> None:
        _get_m(callback_names="mlflow")

    def test_disable_logging(self) -> None:
        m = _get_m()
        state = m.trainer.state
        self.assertTrue(state.enable_logging)
        with state.disable_logging:
            self.assertFalse(state.enable_logging)
        self.assertTrue(state.enable_logging)

    def test_loss_protocol(self) -> None:
        def _get_loss(loss_ins: cflearn.LossProtocol) -> float:
            return loss_ins(forward_results, batch)[cflearn.LOSS_KEY].item()

        forward_results = {cflearn.PREDICTIONS_KEY: torch.full([10, 1], 2.0)}
        batch = {cflearn.LABEL_KEY: torch.zeros([10, 1])}
        self.assertEqual(_get_loss(cflearn.MAELoss()), 2.0)
        self.assertEqual(_get_loss(cflearn.MAELoss(reduction="sum")), 20.0)
        # multi task
        multi_task_name = cflearn.LossProtocol.parse("multi_task:mae,mse")
        multi_task_loss = cflearn.LossProtocol.make(multi_task_name, {})
        losses = multi_task_loss(forward_results, batch)
        self.assertEqual(losses[cflearn.LOSS_KEY].item(), 6.0)
        self.assertEqual(losses["mae"].item(), 2.0)
        self.assertEqual(losses["mse"].item(), 4.0)
        # with auxiliary key
        another_key = "another_key"
        with_aux_name = cflearn.LossProtocol.parse(f"mae:aux:{another_key}")
        with_aux = cflearn.LossProtocol.make(with_aux_name, {})
        forward_results[another_key] = forward_results[cflearn.PREDICTIONS_KEY]
        batch[another_key] = batch[cflearn.LABEL_KEY]
        losses = with_aux(forward_results, batch)
        self.assertEqual(losses[cflearn.LOSS_KEY].item(), 4.0)
        self.assertEqual(losses[cflearn.AuxLoss.main_loss_key].item(), 2.0)
        self.assertEqual(losses[another_key].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
