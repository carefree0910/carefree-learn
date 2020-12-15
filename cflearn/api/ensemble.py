import os
import math
import torch

import numpy as np

from typing import *
from tqdm.autonotebook import tqdm
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cfdata.tabular import task_type_type
from cftool.ml.utils import collate_fn_type
from cftool.ml.utils import Metrics
from torch.nn.functional import one_hot

from .basic import *
from ..misc.toolkit import *
from .register import register_metric
from ..types import data_type
from ..pipeline import Pipeline
from ..protocol import DataProtocol


class EnsembleResults(NamedTuple):
    data: DataProtocol
    pipelines: List[Pipeline]
    pattern_weights: Optional[np.ndarray]
    predict_config: Optional[Dict[str, Any]]

    @property
    def pattern(self) -> EnsemblePattern:
        predict_config = self.predict_config or {}
        patterns = [m.to_pattern(**predict_config) for m in self.pipelines]
        return Ensemble.stacking(patterns, pattern_weights=self.pattern_weights)


class MetricsPlaceholder(NamedTuple):
    config: Dict[str, Any]


class Ensemble:
    def __init__(
        self,
        task_type: task_type_type,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.task_type = task_type
        if config is None:
            config = {}
        self.config = shallow_copy_dict(config)

    @staticmethod
    def stacking(
        patterns: List[ModelPattern],
        *,
        pattern_weights: Optional[np.ndarray] = None,
        ensemble_method: Optional[Union[str, collate_fn_type]] = None,
    ) -> EnsemblePattern:
        if ensemble_method is None:
            if pattern_weights is None:
                ensemble_method = "default"
            else:
                if abs(pattern_weights.sum() - 1.0) > 1e-4:
                    raise ValueError("`pattern_weights` should sum to 1.0")
                pattern_weights = pattern_weights.reshape([-1, 1, 1])

                def ensemble_method(
                    arrays: List[np.ndarray],
                    requires_prob: bool,
                ) -> np.ndarray:
                    shape = [len(arrays), len(arrays[0]), -1]
                    predictions = np.array(arrays).reshape(shape)
                    if requires_prob or not is_int(predictions):
                        return (predictions * pattern_weights).sum(axis=0)
                    encodings = one_hot(to_torch(predictions).to(torch.long).squeeze())
                    encodings = encodings.to(torch.float32)
                    weighted = (encodings * pattern_weights).sum(dim=0)
                    return to_numpy(weighted.argmax(1)).reshape([-1, 1])

        return EnsemblePattern(patterns, ensemble_method)

    def bagging(
        self,
        x: data_type,
        y: data_type = None,
        *,
        k: int = 10,
        num_jobs: int = 1,
        model: str = "fcnn",
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        predict_config: Optional[Dict[str, Any]] = None,
        sequential: Optional[bool] = None,
        temp_folder: str = "__tmp__",
        return_patterns: bool = True,
        use_tqdm: bool = True,
    ) -> EnsembleResults:
        repeat_result = repeat_with(
            x,
            y,
            models=model,
            model_configs=model_configs,
            predict_config=predict_config,
            sequential=sequential,
            num_jobs=num_jobs,
            num_repeat=k,
            temp_folder=temp_folder,
            return_patterns=return_patterns,
            use_tqdm=use_tqdm,
            **self.config,
        )

        data = repeat_result.data
        pipelines = repeat_result.pipelines
        assert data is not None and pipelines is not None
        return EnsembleResults(data, pipelines[model], None, predict_config)

    def adaboost(
        self,
        x: data_type,
        y: data_type = None,
        *,
        k: int = 10,
        eps: float = 1e-12,
        model: str = "fcnn",
        temp_folder: str = "__tmp__",
        predict_config: Optional[Dict[str, Any]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> EnsembleResults:
        if increment_config is None:
            increment_config = {}
        config = shallow_copy_dict(self.config)
        update_dict(increment_config, config)
        config["cv_split"] = 0.0
        config.setdefault("use_tqdm", False)
        config.setdefault("use_binary_threshold", False)
        config.setdefault("verbose_level", 0)

        @register_metric("adaboost_error", -1, False)
        def adaboost_error(
            self_: Union[Metrics, MetricsPlaceholder],
            target_: np.ndarray,
            predictions_: np.ndarray,
        ) -> float:
            target_ = target_.astype(np.float32)
            predictions_ = predictions_.astype(np.float32)
            sample_weights_ = self_.config.get("sample_weights")
            errors = (target_ != predictions_).ravel()
            if sample_weights_ is None:
                e_ = errors.mean()
            else:
                e_ = sample_weights_[errors].sum() / len(errors)
            return e_.item()

        data = None
        pipelines = []
        patterns, pattern_weights = [], []
        for i in tqdm(list(range(k))):
            cfg = shallow_copy_dict(config)
            cfg["logging_folder"] = os.path.join(temp_folder, str(i))
            metric_config = {"sample_weights": sample_weights}
            if sample_weights is not None:
                cfg["metric_config"] = {
                    "types": "adaboost_error",
                    "adaboost_error_config": metric_config,
                }
            m = make(model=model, **cfg)
            m.fit(x, y, sample_weights=sample_weights)
            metrics_placeholder = MetricsPlaceholder(metric_config)
            predictions: np.ndarray = m.predict(x, contains_labels=True)
            predictions = predictions.astype(np.float32)
            target = m.data.processed.y.astype(np.float32)
            e = adaboost_error(metrics_placeholder, target, predictions)
            em = min(max(e, eps), 1.0 - eps)
            am = 0.5 * math.log(1.0 / em - 1.0)
            if sample_weights is None:
                sample_weights = np.ones_like(predictions).ravel()
            target[target == 0.0] = predictions[predictions == 0.0] = -1.0
            sample_weights *= np.exp(-am * target * predictions).ravel()
            sample_weights /= np.mean(sample_weights)
            patterns.append(m.to_pattern())
            pattern_weights.append(am)
            if data is None:
                data = m.data
            pipelines.append(m)

        weights_array = np.array(pattern_weights, np.float32)
        weights_array /= weights_array.sum()

        assert data is not None
        return EnsembleResults(data, pipelines, weights_array, predict_config)


__all__ = [
    "Ensemble",
    "EnsembleResults",
]
