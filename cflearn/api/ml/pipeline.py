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
from cftool.misc import get_arguments
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.array import softmax
from cftool.array import squeeze
from cftool.array import is_float
from cftool.array import get_full_logits
from cftool.array import get_label_predictions
from cftool.types import np_dict_type

from ...data import MLData
from ...data import MLLoader
from ...data import MLDataset
from ...data import DLDataModule
from ...data import MLInferenceData
from ...types import configs_type
from ...types import states_callback_type
from ...trainer import get_sorted_checkpoints
from ...pipeline import DLPipeline
from ...protocol import InferenceOutputs
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import PREDICTIONS_KEY
from ...misc.internal_.inference import MLInference
from ...models.ml.encoders import Encoder
from ...models.ml.encoders import EncodingSettings
from ...models.protocols.ml import MLModel

try:
    from cfdata.tabular import ColumnTypes
    from cfml.misc.toolkit import ModelPattern
except:
    ColumnTypes = ModelPattern = None


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
        use_encoder_cache: bool = True,
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_settings: Optional[Dict[int, Dict[str, Any]]] = None,
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
        cudnn_benchmark: bool = False,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: configs_type = None,
        metric_weights: Optional[Dict[str, float]] = None,
        use_losses_as_metrics: Optional[bool] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        recompute_train_losses_in_eval: bool = True,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        update_scheduler_per_epoch: bool = False,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        use_zero: bool = False,
        workplace: str = "_logs",
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
            cudnn_benchmark=cudnn_benchmark,
            metric_names=metric_names,
            metric_configs=metric_configs,
            metric_weights=metric_weights,
            use_losses_as_metrics=use_losses_as_metrics,
            loss_metrics_weights=loss_metrics_weights,
            recompute_train_losses_in_eval=recompute_train_losses_in_eval,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            lr=lr,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            update_scheduler_per_epoch=update_scheduler_per_epoch,
            optimizer_settings=optimizer_settings,
            use_zero=use_zero,
            workplace=workplace,
            finetune_config=finetune_config,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
        )
        self.core_name = core_name
        self.core_config = core_config or {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_encoder_cache = use_encoder_cache
        self.only_categorical = only_categorical
        self.encoder = None
        self.encoder_config = encoder_config or {}
        self.encoding_settings = {}
        for idx, raw_setting in (encoding_settings or {}).items():
            self.encoding_settings[idx] = EncodingSettings(**raw_setting)
        self._pre_process_batch = pre_process_batch
        self._num_repeat = num_repeat

    def _setup_defaults(self, data_info: Dict[str, Any]) -> None:
        self.is_classification = data_info["is_classification"]
        is_reg = not self.is_classification
        if self.input_dim is None:
            self.input_dim = data_info["input_dim"]
        if self.output_dim is None:
            self.output_dim = 1 if is_reg else data_info["num_classes"]
            if self.output_dim is None:
                raise ValueError(
                    "either `MLData` with `cf_data`, or `output_dim`, "
                    "should be provided"
                )
        self.use_auto_loss = self.loss_name == "auto"
        if self.use_auto_loss:
            if is_reg:
                self.loss_name = "mae"
            else:
                self.loss_name = "bce" if self.output_dim == 1 else "focal"

    def _instantiate_encoder(self, settings: Dict[int, EncodingSettings]) -> Encoder:
        if self.in_loading or not self.use_encoder_cache:
            loaders = []
        else:
            train_loader, valid_loader = self.data.initialize()
            train_loader_copy = train_loader.copy()
            train_loader_copy.disable_shuffle()
            loaders = [train_loader_copy]
            if valid_loader is not None:
                assert isinstance(valid_loader, MLLoader)
                loaders.append(valid_loader)
        return Encoder(settings, config=self.encoder_config, loaders=loaders)

    def _setup_encoder(self, data_info: Dict[str, Any]) -> None:
        assert isinstance(self.input_dim, int)
        all_indices = list(range(self.input_dim))
        if not self.encoding_settings:
            self.encoder = None
            self.numerical_columns_mapping = {i: i for i in all_indices}
            self.categorical_columns_mapping = {}
            self.use_one_hot = False
            self.use_embedding = False
        else:
            use_one_hot = False
            use_embedding = False
            for idx, setting in self.encoding_settings.items():
                use_one_hot = use_one_hot or setting.use_one_hot
                use_embedding = use_embedding or setting.use_embedding
            self.encoder = self._instantiate_encoder(self.encoding_settings)
            self.use_one_hot = use_one_hot
            self.use_embedding = use_embedding
            categorical_columns = self.encoder.columns.copy()
            categorical_set = set(categorical_columns)
            numerical_columns = [i for i in all_indices if i not in categorical_set]
            self.numerical_columns_mapping = {i: i for i in numerical_columns}
            self.categorical_columns_mapping = {i: i for i in categorical_columns}

    def _prepare_modules(self, data_info: Dict[str, Any]) -> None:
        self._setup_defaults(data_info)
        self._setup_encoder(data_info)
        self._prepare_workplace()
        self._prepare_loss()
        assert isinstance(self.input_dim, int)
        assert isinstance(self.output_dim, int)
        self.model = MLModel(
            self.output_dim,
            data_info["num_history"],
            encoder=self.encoder,
            use_encoder_cache=self.use_encoder_cache,
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

    def _prepare_trainer_defaults(self, data_info: Dict[str, Any]) -> None:
        if self.trainer_config["monitor_names"] is None:
            self.trainer_config["monitor_names"] = ["mean_std", "plateau"]
        super()._prepare_trainer_defaults(data_info)
        if (
            self.trainer_config["metric_names"] is None
            and self.use_auto_loss
            and not self.trainer_config["use_losses_as_metrics"]
        ):
            if data_info["is_classification"]:
                self.trainer_config["metric_names"] = ["acc", "auc"]
            else:
                self.trainer_config["metric_names"] = ["mae", "mse"]
        callback_names = self.trainer_config["callback_names"]
        if "_inject_loader_name" not in callback_names:
            callback_names.append("_inject_loader_name")

    def _make_new_loader(  # type: ignore
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
        binary_threshold: float = 0.5,
        pre_process: Optional[Callable] = None,
        **predict_kwargs: Any,
    ) -> ModelPattern:
        if ModelPattern is None:
            raise ValueError("`carefree-ml` need to be installed to use `to_pattern`")

        def _predict(data: MLInferenceData) -> np.ndarray:
            if pre_process is not None:
                data = pre_process(data)
            predictions = self.predict(data, **predict_kwargs)[PREDICTIONS_KEY]
            if self.is_classification:
                return get_label_predictions(predictions, binary_threshold)
            return predictions

        def _predict_prob(data: MLInferenceData) -> np.ndarray:
            if not self.is_classification:
                msg = "`predict_prob` should not be called in regression tasks"
                raise ValueError(msg)
            if pre_process is not None:
                data = pre_process(data)
            logits = self.predict(data, **predict_kwargs)[PREDICTIONS_KEY]
            logits = get_full_logits(logits)
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
        post_callback: Optional[Callable[[DLPipeline, Dict[str, Any]], None]] = None,
    ) -> "SimplePipeline":
        export_folder = export_folders[0]
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                m = cls._load_infrastructure(
                    export_folder,
                    cuda,
                    False,
                    pre_callback,
                    post_callback,
                )
                assert isinstance(m, SimplePipeline)
                data_info = DLDataModule.load_info(export_folder)
        m._num_repeat = m.config["num_repeat"] = len(export_folders)
        m._prepare_modules(data_info)
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
            self._save_misc(export_folder)
            file = f"{PT_PREFIX}-1.pt"
            torch.save(self.model.state_dict(), os.path.join(export_folder, file))
            with open(os.path.join(export_folder, SCORES_FILE), "w") as f:
                json.dump({file: 0.0}, f)
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

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


@DLPipeline.register("ml.carefree")
class CarefreePipeline(SimplePipeline):
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
        use_encoder_cache: bool = True,
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_settings: Optional[Dict[int, Dict[str, Any]]] = None,
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
        cudnn_benchmark: bool = False,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: configs_type = None,
        metric_weights: Optional[Dict[str, float]] = None,
        use_losses_as_metrics: Optional[bool] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        recompute_train_losses_in_eval: bool = True,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        update_scheduler_per_epoch: bool = False,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        use_zero: bool = False,
        workplace: str = "_logs",
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
        pre_process_batch: bool = True,
        num_repeat: Optional[int] = None,
    ):
        if ColumnTypes is None:
            raise ValueError(
                "`carefree-data` need to be installed for `ml.CarefreePipeline`, "
                "please use `ml.SimplePipeline` instead."
            )
        config = get_arguments()
        super().__init__(
            core_name,
            core_config,
            input_dim=input_dim,
            output_dim=output_dim,
            loss_name=loss_name,
            loss_config=loss_config,
            use_encoder_cache=use_encoder_cache,
            only_categorical=only_categorical,
            encoder_config=encoder_config,
            encoding_settings=encoding_settings,
            state_config=state_config,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            fixed_epoch=fixed_epoch,
            fixed_steps=fixed_steps,
            log_steps=log_steps,
            valid_portion=valid_portion,
            amp=amp,
            clip_norm=clip_norm,
            cudnn_benchmark=cudnn_benchmark,
            metric_names=metric_names,
            metric_configs=metric_configs,
            metric_weights=metric_weights,
            use_losses_as_metrics=use_losses_as_metrics,
            loss_metrics_weights=loss_metrics_weights,
            recompute_train_losses_in_eval=recompute_train_losses_in_eval,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            lr=lr,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            update_scheduler_per_epoch=update_scheduler_per_epoch,
            optimizer_settings=optimizer_settings,
            use_zero=use_zero,
            workplace=workplace,
            finetune_config=finetune_config,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
            pre_process_batch=pre_process_batch,
            num_repeat=num_repeat,
        )
        self.config = config

    def _setup_encoder(self, data_info: Dict[str, Any]) -> None:
        self.cf_data = data_info["cf_data"]
        if self.cf_data is None:
            msg = "cf_data` is not provided, please use `ml.SimplePipeline` instead"
            raise ValueError(msg)
        # encoder
        excluded = 0
        encoder_settings = {}
        numerical_columns_mapping = {}
        categorical_columns_mapping = {}
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
                elif idx not in encoder_settings:
                    true_idx = idx - excluded
                    setting = EncodingSettings(dim=recognizer.num_unique_values)
                    encoder_settings[true_idx] = setting
                    use_one_hot = use_one_hot or setting.use_one_hot
                    use_embedding = use_embedding or setting.use_embedding
                    categorical_columns_mapping[idx] = true_idx
        if not encoder_settings:
            encoder = None
        else:
            encoder = self._instantiate_encoder(encoder_settings)
        self.encoder = encoder
        self.use_one_hot = use_one_hot
        self.use_embedding = use_embedding
        self.numerical_columns_mapping = numerical_columns_mapping
        self.categorical_columns_mapping = categorical_columns_mapping

    def _predict_from_outputs(self, outputs: InferenceOutputs) -> np_dict_type:
        assert self.cf_data is not None
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
                    v = squeeze(np.apply_along_axis(fn, axis=0, arr=v))
            recovered[k] = v
        return recovered

    def _make_new_loader(  # type: ignore
        self,
        data: MLData,
        batch_size: int,
        **kwargs: Any,
    ) -> MLLoader:
        assert self.cf_data is not None
        return MLLoader(
            MLDataset(*self.cf_data.transform(data.x_train, None, **kwargs).xy),
            shuffle=False,
            batch_size=batch_size,
        )

    @classmethod
    def _load_infrastructure(
        cls,
        export_folder: str,
        cuda: Optional[str],
        to_original_device: bool,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[[DLPipeline, Dict[str, Any]], None]] = None,
    ) -> "CarefreePipeline":
        m = super()._load_infrastructure(
            export_folder,
            cuda,
            to_original_device,
            pre_callback,
            post_callback,
        )
        assert isinstance(m, CarefreePipeline)
        m.cf_data = DLDataModule.load_info(export_folder)["cf_data"]
        return m


__all__ = [
    "SimplePipeline",
    "CarefreePipeline",
]
