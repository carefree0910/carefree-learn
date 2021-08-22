import os
import json
import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from functools import partial
from collections import OrderedDict
from cfdata.tabular import ColumnTypes
from cfdata.tabular import TabularData
from cftool.ml import ModelPattern
from cftool.misc import lock_manager
from cftool.misc import Saving

from .data import MLData
from ...types import np_dict_type
from ...types import states_callback_type
from ...trainer import get_sorted_checkpoints
from ...protocol import InferenceOutputs
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import PREDICTIONS_KEY
from ..internal_.pipeline import DLPipeline
from ...misc.toolkit import softmax
from ...misc.toolkit import is_float
from ...misc.toolkit import get_arguments
from ...misc.internal_ import MLLoader
from ...misc.internal_ import MLDataset
from ...misc.internal_ import MLInference
from ...models.ml.encoders import Encoder
from ...models.ml.protocol import MLModel


@DLPipeline.register("ml.simple")
class SimplePipeline(DLPipeline):
    model: MLModel
    inference: MLInference
    inference_base = MLInference

    encoder: Optional[Encoder]
    numerical_columns_mapping: Dict[int, int]
    categorical_columns_mapping: Dict[int, int]
    use_one_hot: bool
    use_embedding: bool

    def __init__(
        self,
        core_name: str = "fcnn",
        core_config: Optional[Dict[str, Any]] = None,
        *,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        loss_name: str = "auto",
        loss_config: Optional[Dict[str, Any]] = None,
        # encoder
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_methods: Optional[Dict[str, List[str]]] = None,
        encoding_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_encoding_methods: Optional[List[str]] = None,
        default_encoding_configs: Optional[Dict[str, Any]] = None,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        fixed_epoch: Optional[int] = None,
        fixed_steps: Optional[int] = None,
        log_steps: Optional[int] = None,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        use_losses_as_metrics: Optional[bool] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        ddp_config: Optional[Dict[str, Any]] = None,
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
        pre_process_batch: bool = True,
        num_repeat: Optional[int] = None,
    ):
        self.config = get_arguments()
        super().__init__(
            loss_name=loss_name,
            loss_config=loss_config,
            state_config=state_config,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            fixed_epoch=fixed_epoch,
            fixed_steps=fixed_steps,
            log_steps=log_steps,
            valid_portion=valid_portion,
            amp=amp,
            clip_norm=clip_norm,
            metric_names=metric_names,
            metric_configs=metric_configs,
            use_losses_as_metrics=use_losses_as_metrics,
            loss_metrics_weights=loss_metrics_weights,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            lr=lr,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            optimizer_settings=optimizer_settings,
            workplace=workplace,
            ddp_config=ddp_config,
            finetune_config=finetune_config,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
        )
        self.core_name = core_name
        self.core_config = core_config or {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.only_categorical = only_categorical
        self.encoder = None
        self.encoder_config = encoder_config or {}
        self.encoding_methods = encoding_methods or {}
        self.encoding_configs = encoding_configs or {}
        if default_encoding_methods is None:
            default_encoding_methods = ["embedding"]
        self.default_encoding_methods = default_encoding_methods
        self.default_encoding_configs = default_encoding_configs or {}
        self._pre_process_batch = pre_process_batch
        self._num_repeat = num_repeat

    def _prepare_modules(self, data_json: Dict[str, Any]) -> None:
        self.is_classification = data_json["is_classification"]
        is_reg = not self.is_classification
        if self.input_dim is None:
            self.input_dim = data_json["input_dim"]
        if self.output_dim is None:
            self.output_dim = 1 if is_reg else data_json["num_classes"]
            if self.output_dim is None:
                raise ValueError(
                    "either `MLData` with `cf_data`, or `output_dim`, "
                    "should be provided"
                )
        self.use_auto_loss = self.loss_name == "auto"
        if self.use_auto_loss:
            self.loss_name = "mae" if is_reg else "focal"
        if self.encoder is None:
            self.numerical_columns_mapping = {i: i for i in range(self.input_dim)}
            self.categorical_columns_mapping = {}
            self.use_one_hot = False
            self.use_embedding = False
        self._prepare_workplace()
        self._prepare_loss()
        self.model = MLModel(
            self.input_dim,
            self.output_dim,
            data_json["num_history"],
            encoder=self.encoder,
            numerical_columns_mapping=self.numerical_columns_mapping,
            categorical_columns_mapping=self.categorical_columns_mapping,
            use_one_hot=self.use_one_hot,
            use_embedding=self.use_embedding,
            only_categorical=self.only_categorical,
            core_name=self.core_name,
            core_config=self.core_config,
            pre_process_batch=self._pre_process_batch,
            num_repeat=self._num_repeat,
        )
        self.inference = MLInference(model=self.model)

    def _prepare_trainer_defaults(self, data_json: Dict[str, Any]) -> None:
        super()._prepare_trainer_defaults(data_json)
        if self.trainer_config["metric_names"] is None and self.use_auto_loss:
            if data_json["is_classification"]:
                self.trainer_config["metric_names"] = ["acc", "auc"]
            else:
                self.trainer_config["metric_names"] = ["mae", "mse"]
        callback_names = self.trainer_config["callback_names"]
        if "_inject_loader_name" not in callback_names:
            callback_names.append("_inject_loader_name")

    def _make_new_loader(
        self,
        data: MLData,
        batch_size: int,
        **kwargs: Any,
    ) -> MLLoader:
        x = data.x_train
        return MLLoader(MLDataset(x, None), shuffle=False, batch_size=batch_size)

    def to_pattern(
        self,
        *,
        pre_process: Optional[Callable] = None,
        **predict_kwargs: Any,
    ) -> ModelPattern:
        def _predict(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            predictions = self.predict(x, **predict_kwargs)[PREDICTIONS_KEY]
            if self.is_classification:
                return np.argmax(predictions, axis=1)[..., None]
            return predictions

        def _predict_prob(x: np.ndarray) -> np.ndarray:
            if not self.is_classification:
                msg = "`predict_prob` should not be called in regression tasks"
                raise ValueError(msg)
            if pre_process is not None:
                x = pre_process(x)
            logits = self.predict(x, **predict_kwargs)[PREDICTIONS_KEY]
            return softmax(logits)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    @classmethod
    def fuse_multiple(
        cls,
        export_folders: List[str],
        *,
        cuda: Optional[str] = None,
        compress: bool = True,
        states_callback: states_callback_type = None,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "SimplePipeline":
        export_folder = export_folders[0]
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                m = cls._load_infrastructure(
                    export_folder,
                    cuda,
                    pre_callback,
                    post_callback,
                )
                assert isinstance(m, SimplePipeline)
        m._num_repeat = m.config["num_repeat"] = len(export_folders)
        with open(os.path.join(export_folder, cls.data_json_file), "r") as f:
            m._prepare_modules(json.load(f))
        m.model.to(m.device)
        merged_states: OrderedDict[str, torch.Tensor] = OrderedDict()
        for i, export_folder in enumerate(export_folders):
            base_folder = os.path.dirname(os.path.abspath(export_folder))
            with lock_manager(base_folder, [export_folder]):
                with Saving.compress_loader(export_folder, compress):
                    checkpoints = get_sorted_checkpoints(export_folder)
                    checkpoint_path = os.path.join(export_folder, checkpoints[0])
                    states = torch.load(checkpoint_path, map_location=m.device)
                    current_keys = list(states.keys())
                    for k, v in list(states.items()):
                        split = k.split(".")
                        split.insert(1, str(i))
                        states[".".join(split)] = v
                    for k in current_keys:
                        states.pop(k)
                    if states_callback is not None:
                        states = states_callback(m, states)
                    merged_states.update(states)
        m.model.load_state_dict(merged_states)
        return m

    def re_save(self, export_folder: str, *, compress: bool = True) -> "SimplePipeline":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            self._save_misc(export_folder, False)
            file = f"{PT_PREFIX}-1.pt"
            torch.save(self.model.state_dict(), os.path.join(export_folder, file))
            with open(os.path.join(export_folder, SCORES_FILE), "w") as f:
                json.dump({file: 0.0}, f)
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self


@DLPipeline.register("ml.carefree")
class CarefreePipeline(SimplePipeline):
    def __init__(
        self,
        core_name: str = "fcnn",
        core_config: Optional[Dict[str, Any]] = None,
        *,
        output_dim: Optional[int] = None,
        loss_name: str = "auto",
        loss_config: Optional[Dict[str, Any]] = None,
        # encoder
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_methods: Optional[Dict[str, List[str]]] = None,
        encoding_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_encoding_methods: Optional[List[str]] = None,
        default_encoding_configs: Optional[Dict[str, Any]] = None,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        fixed_epoch: Optional[int] = None,
        fixed_steps: Optional[int] = None,
        log_steps: Optional[int] = None,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        use_losses_as_metrics: Optional[bool] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        ddp_config: Optional[Dict[str, Any]] = None,
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
        num_repeat: Optional[int] = None,
    ):
        config = get_arguments()
        super().__init__(
            core_name,
            core_config,
            output_dim=output_dim,  # type: ignore
            loss_name=loss_name,
            loss_config=loss_config,
            only_categorical=only_categorical,
            encoder_config=encoder_config,
            encoding_methods=encoding_methods,
            encoding_configs=encoding_configs,
            default_encoding_methods=default_encoding_methods,
            default_encoding_configs=default_encoding_configs,
            state_config=state_config,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            fixed_epoch=fixed_epoch,
            fixed_steps=fixed_steps,
            log_steps=log_steps,
            valid_portion=valid_portion,
            amp=amp,
            clip_norm=clip_norm,
            metric_names=metric_names,
            metric_configs=metric_configs,
            use_losses_as_metrics=use_losses_as_metrics,
            loss_metrics_weights=loss_metrics_weights,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            lr=lr,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            optimizer_settings=optimizer_settings,
            workplace=workplace,
            ddp_config=ddp_config,
            finetune_config=finetune_config,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
            num_repeat=num_repeat,
        )
        self.config = config
        self.cf_data = None

    def _prepare_modules(self, data_json: Dict[str, Any]) -> None:
        if self.cf_data is None:
            self.cf_data = self.data.cf_data
        # encoder
        excluded = 0
        numerical_columns_mapping = {}
        categorical_columns_mapping = {}
        categorical_dims = []
        encoding_methods = []
        encoding_configs = []
        true_categorical_columns = []
        use_one_hot = False
        use_embedding = False
        if self.cf_data.is_simplify:
            for idx in range(self.cf_data.processed.x.shape[1]):
                numerical_columns_mapping[idx] = idx
        else:
            ts_indices = self.cf_data.ts_indices
            recognizers = self.cf_data.recognizers
            sorted_indices = [idx for idx in sorted(recognizers) if idx != -1]
            for idx in sorted_indices:
                recognizer = recognizers[idx]
                assert recognizer is not None
                if not recognizer.info.is_valid or idx in ts_indices:
                    excluded += 1
                elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                    numerical_columns_mapping[idx] = idx - excluded
                else:
                    str_idx = str(idx)
                    categorical_dims.append(recognizer.num_unique_values)
                    idx_encoding_methods = self.encoding_methods.setdefault(
                        str_idx,
                        self.default_encoding_methods,
                    )
                    if isinstance(idx_encoding_methods, str):
                        idx_encoding_methods = [idx_encoding_methods]
                    use_one_hot = use_one_hot or "one_hot" in idx_encoding_methods
                    use_embedding = use_embedding or "embedding" in idx_encoding_methods
                    encoding_methods.append(idx_encoding_methods)
                    encoding_configs.append(
                        self.encoding_configs.setdefault(
                            str_idx,
                            self.default_encoding_configs,
                        )
                    )
                    true_idx = idx - excluded
                    true_categorical_columns.append(true_idx)
                    categorical_columns_mapping[idx] = true_idx
        if not true_categorical_columns:
            encoder = None
        else:
            if self.in_loading:
                loaders = []
            else:
                train_loader, valid_loader = self.data.initialize()
                train_loader_copy = train_loader.copy()
                train_loader_copy.disable_shuffle()
                loaders = [train_loader_copy]
                if valid_loader is not None:
                    loaders.append(valid_loader)
            encoder = Encoder(
                self.encoder_config,
                categorical_dims,
                encoding_methods,  # type: ignore
                encoding_configs,
                true_categorical_columns,
                loaders,
            )
        self.encoder = encoder
        self.use_one_hot = use_one_hot
        self.use_embedding = use_embedding
        self.numerical_columns_mapping = numerical_columns_mapping
        self.categorical_columns_mapping = categorical_columns_mapping
        super()._prepare_modules(data_json)

    def _predict_from_outputs(self, outputs: InferenceOutputs) -> np_dict_type:
        results = outputs.forward_results
        if self.cf_data.is_clf:
            return results
        fn = partial(self.cf_data.recover_labels, inplace=True)
        recovered = {}
        for k, v in results.items():
            if is_float(v):
                if v.shape[1] == 1:
                    v = fn(v)
                else:
                    v = np.apply_along_axis(fn, axis=0, arr=v).squeeze()
            recovered[k] = v
        return recovered

    def _make_new_loader(
        self,
        data: MLData,
        batch_size: int,
        **kwargs: Any,
    ) -> MLLoader:
        return MLLoader(
            MLDataset(*self.cf_data.transform(data.x_train, None, **kwargs).xy),
            shuffle=False,
            batch_size=batch_size,
        )

    def _save_misc(self, export_folder: str, retain_data: bool) -> float:
        data_folder = os.path.join(export_folder, self.data_folder)
        self.cf_data.save(data_folder, retain_data=retain_data, compress=False)
        return super()._save_misc(export_folder, retain_data)

    @classmethod
    def _load_infrastructure(
        cls,
        export_folder: str,
        cuda: Optional[str],
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "CarefreePipeline":
        m = super()._load_infrastructure(
            export_folder,
            cuda,
            pre_callback,
            post_callback,
        )
        assert isinstance(m, CarefreePipeline)
        data_folder = os.path.join(export_folder, cls.data_folder)
        m.cf_data = TabularData.load(data_folder, compress=False)
        return m

    @classmethod
    def _load_states_callback(cls, m: Any, states: Dict[str, Any]) -> Dict[str, Any]:
        if m.encoder is not None:
            encoder_cache_keys = []
            for key in states:
                if key.startswith("encoder") and key.endswith("cache"):
                    encoder_cache_keys.append(key)
            for key in encoder_cache_keys:
                states.pop(key)
        return states

    @classmethod
    def pack(
        cls,
        workplace: str,
        *,
        input_dim: Optional[int] = None,
        config_bundle_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        pack_folder: Optional[str] = None,
        cuda: Optional[str] = None,
    ) -> str:
        raise ValueError(
            "`CarefreePipeline` does not support packing from workplace, "
            "because it utilizes `carefree-data` which cannot be accessed"
        )


__all__ = [
    "SimplePipeline",
    "CarefreePipeline",
]
