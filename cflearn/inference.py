import numpy as np

from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import filter_kw
from cftool.misc import shallow_copy_dict
from cftool.array import to_numpy
from cftool.array import to_device
from cftool.types import np_dict_type

from .schema import ILoss
from .schema import IMetric
from .schema import IDLModel
from .schema import IInference
from .schema import IDataLoader
from .schema import TrainerState
from .schema import MetricsOutputs
from .schema import InferenceOutputs
from .constants import INPUT_KEY
from .constants import LABEL_KEY
from .constants import BATCH_INDICES_KEY
from .constants import ORIGINAL_LABEL_KEY
from .misc.toolkit import eval_context
from .misc.toolkit import get_device
from .misc.toolkit import np_batch_to_tensor
from .misc.toolkit import ONNX


@IInference.register("dl")
class DLInference(IInference):
    def __init__(
        self,
        *,
        onnx: Optional[ONNX] = None,
        model: Optional[IDLModel] = None,
        use_grad_in_predict: bool = False,
    ):
        self.onnx = onnx
        self.model = model
        if onnx is None and model is None:
            raise ValueError("either `onnx` or `model` should be provided")
        if onnx is not None and model is not None:
            raise ValueError("only one of `onnx` & `model` should be provided")
        self.use_grad_in_predict = use_grad_in_predict

    def get_outputs(
        self,
        loader: IDataLoader,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        metrics: Optional[IMetric] = None,
        loss: Optional[ILoss] = None,
        return_outputs: bool = True,
        stack_outputs: bool = True,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> InferenceOutputs:
        def _core() -> InferenceOutputs:
            results: Dict[str, Optional[List[np.ndarray]]] = {}
            metric_outputs_list: List[MetricsOutputs] = []
            loss_items: Dict[str, List[float]] = {}
            labels: Dict[str, List[np.ndarray]] = {}
            iterator = enumerate(loader)
            if use_tqdm:
                iterator = tqdm(iterator, "inference", len(loader))
            requires_all_outputs = return_outputs
            if metrics is not None and metrics.requires_all:
                requires_all_outputs = True
            for i, np_batch in iterator:
                if i / len(loader) >= portion:
                    break
                batch = None
                if self.model is not None or loss is not None:
                    batch = np_batch_to_tensor(np_batch)
                    if self.model is not None:
                        batch = to_device(batch, get_device(self.model))
                if self.onnx is not None:
                    local_outputs = np_batch_to_tensor(self.onnx.predict(np_batch))
                else:
                    assert self.model is not None
                    with eval_context(self.model, use_grad=use_grad):
                        assert not self.model.training
                        kw = filter_kw(self.model.forward, shallow_copy_dict(kwargs))
                        local_outputs = self.model.run(i, batch, state, **kw)
                # gather outputs
                requires_metrics = metrics is not None and not metrics.requires_all
                requires_np = requires_metrics or requires_all_outputs
                np_outputs: np_dict_type = {}
                for k, v in local_outputs.items():
                    if not requires_np:
                        results[k] = None
                        continue
                    if v is None:
                        continue
                    if isinstance(v, np.ndarray):
                        v_np = v
                    elif isinstance(v, Tensor):
                        v_np = to_numpy(v)
                    elif isinstance(v, list):
                        if isinstance(v[0], np.ndarray):
                            v_np = v
                        else:
                            v_np = list(map(to_numpy, v))
                    else:
                        raise ValueError(f"unrecognized value ({k}={type(v)}) occurred")
                    np_outputs[k] = v_np
                    if not requires_all_outputs:
                        results[k] = None
                    else:
                        results.setdefault(k, []).append(v_np)  # type: ignore
                if requires_np:
                    for k, v in np_batch.items():
                        if (
                            k == INPUT_KEY
                            or k == ORIGINAL_LABEL_KEY
                            or k.endswith(BATCH_INDICES_KEY)
                        ):
                            continue
                        if v is None:
                            continue
                        if k != LABEL_KEY and len(v.shape) > 2:
                            continue
                        labels.setdefault(k, []).append(v)
                # metrics
                if requires_metrics:
                    sub_outputs = metrics.evaluate(np_batch, np_outputs, loader)  # type: ignore
                    metric_outputs_list.append(sub_outputs)
                # loss
                if loss is not None:
                    with eval_context(loss, use_grad=use_grad):
                        local_losses = loss.run(local_outputs, batch)
                    for k, v in local_losses.items():
                        loss_items.setdefault(k, []).append(v.item())
            # gather outputs
            final_results: Dict[str, Union[np.ndarray, Any]]
            if not requires_all_outputs:
                final_results = {k: None for k in results}
            else:
                final_results = {
                    batch_key: batch_results
                    if not stack_outputs
                    else np.vstack(batch_results)
                    if isinstance(batch_results[0], np.ndarray)
                    else [
                        np.vstack([batch[i] for batch in batch_results])
                        for i in range(len(batch_results[0]))
                    ]
                    for batch_key, batch_results in results.items()
                    if batch_results is not None
                }
            # gather metric outputs
            if metrics is None:
                metric_outputs = None
            elif metrics.requires_all:
                metric_outputs = metrics.evaluate(
                    {k: np.vstack(v) for k, v in labels.items()},
                    final_results,
                    loader,
                )
            else:
                scores = []
                metric_values: Dict[str, List[float]] = {}
                is_positive: Dict[str, bool] = {}
                for sub_outputs in metric_outputs_list:
                    scores.append(sub_outputs.final_score)
                    for k, v in sub_outputs.metric_values.items():
                        metric_values.setdefault(k, []).append(v)
                        existing_is_positive = is_positive.get(k)
                        k_is_positive = sub_outputs.is_positive[k]
                        if (
                            existing_is_positive is not None
                            and existing_is_positive != k_is_positive
                        ):
                            raise ValueError(
                                f"the `is_positive` property of '{k}' collides: "
                                f"{existing_is_positive} (previous) != {k_is_positive}"
                            )
                        is_positive[k] = k_is_positive
                metric_outputs = MetricsOutputs(
                    sum(scores) / len(scores),
                    {k: sum(vl) / len(vl) for k, vl in metric_values.items()},
                    is_positive,
                )

            target_labels = labels.get(LABEL_KEY, [])
            return InferenceOutputs(
                final_results,
                None if not target_labels else np.vstack(target_labels),
                metric_outputs,
                None
                if not loss_items
                else {k: sum(v) / len(v) for k, v in loss_items.items()},
            )

        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        with loader.temporarily_disable_shuffle():
            try:
                return _core()
            except:
                use_grad = self.use_grad_in_predict = True
                return _core()


__all__ = [
    "DLInference",
]
