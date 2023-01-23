import os
import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import Protocol
from collections import OrderedDict
from cftool.misc import lock_manager
from cftool.misc import safe_execute
from cftool.misc import Saving
from cftool.array import softmax
from cftool.array import get_full_logits
from cftool.array import get_label_predictions
from cftool.types import np_dict_type

from ...data import MLData
from ...data import MLLoader
from ...data import DataModule
from ...types import data_type
from ...types import configs_type
from ...types import sample_weights_type
from ...types import states_callback_type
from ...schema import InferenceOutputs
from ...trainer import get_sorted_checkpoints
from ...pipeline import get_requirements
from ...pipeline import IModifier
from ...pipeline import DLPipeline
from ...constants import PREDICTIONS_KEY
from ...data.ml import IMLData
from ...data.ml import MLCarefreeData
from ...data.ml import IMLDataProcessor
from ...data.ml import _InternalCarefreeMLDataProcessor
from ...misc.toolkit import ConfigMeta
from ...misc.internal_.inference import MLInference
from ...models.ml.encoders import Encoder
from ...models.ml.encoders import EncodingSettings
from ...models.schemas.ml import MLModel

try:
    from cfdata.tabular import ColumnTypes
    from cfdata.tabular import TabularData
    from cfml.misc.toolkit import ModelPattern
except:
    ColumnTypes = TabularData = ModelPattern = None


class IMLPredict(Protocol):
    def __call__(
        self,
        data: IMLData,
        *,
        return_classes: bool = False,
        binary_threshold: float = 0.5,
        return_probabilities: bool = False,
        contains_labels: bool = True,
        **predict_kwargs: Any,
    ) -> np_dict_type:
        pass


class IMLMakeModifier(Protocol):
    def __call__(self) -> "MLModifier":
        pass


class IMLMakeInferenceData(Protocol):
    def __call__(
        self,
        x: data_type,
        y: data_type = None,
        *,
        shuffle: bool = False,
        sample_weights: sample_weights_type = None,
    ) -> IMLData:
        pass


class IMLPipeline:
    processor: IMLDataProcessor

    is_classification: bool
    output_dim: Optional[int]
    use_auto_loss: bool

    encoder: Optional[Encoder]
    encoder_config: Dict[str, Any]
    encoding_settings: Dict[int, EncodingSettings]
    numerical_columns: List[int]
    categorical_columns: List[int]
    use_one_hot: bool
    use_embedding: bool
    only_categorical: bool
    use_encoder_cache: bool

    core_name: str
    core_config: Dict[str, Any]

    _pre_process_batch: bool
    _num_repeat: Optional[int]
    _make_modifier: IMLMakeModifier

    predict: IMLPredict
    make_inference_data: IMLMakeInferenceData


_ml_requirements = get_requirements(IMLPipeline, excludes=[])
_ml_build_steps = ["setup_processor", "setup_defaults", "setup_encoder"]


@IModifier.register("ml")
class MLModifier(IModifier, IMLPipeline):
    build_steps = _ml_build_steps + IModifier.build_steps
    requirements = IModifier.requirements + _ml_requirements

    # build steps

    def setup_processor(self, data_info: Dict[str, Any]) -> None:
        self.processor = data_info["processor"]

    def setup_defaults(self, data_info: Dict[str, Any]) -> None:
        self.is_classification = data_info["is_classification"]
        is_reg = not self.is_classification
        if self.input_dim is None:
            self.input_dim = data_info["input_dim"]
            self._defaults["input_dim"] = self.input_dim
        if self.output_dim is None:
            self.output_dim = 1 if is_reg else data_info["num_classes"]
            if self.output_dim is None:
                raise ValueError(
                    "either `MLData` with `cf_data`, or `output_dim`, "
                    "should be provided"
                )
            self._defaults["output_dim"] = self.output_dim
        self.use_auto_loss = self.loss_name == "auto"
        if self.use_auto_loss:
            if is_reg:
                self.loss_name = "mae"
            else:
                self.loss_name = "bce" if self.output_dim == 1 else "focal"
            self._defaults["loss_name"] = self.loss_name

    def setup_encoder(self, data_info: Dict[str, Any]) -> None:
        assert isinstance(self.input_dim, int)
        all_indices = list(range(self.input_dim))
        if not self.encoding_settings:
            self.encoder = None
            self.numerical_columns = all_indices
            self.categorical_columns = []
            self.use_one_hot = False
            self.use_embedding = False
        else:
            use_one_hot = False
            use_embedding = False
            for setting in self.encoding_settings.values():
                use_one_hot = use_one_hot or setting.use_one_hot
                use_embedding = use_embedding or setting.use_embedding
            self.encoder = self._instantiate_encoder(self.encoding_settings)
            self.use_one_hot = use_one_hot
            self.use_embedding = use_embedding
            categorical_columns = self.encoder.columns.copy()
            categorical_set = set(categorical_columns)
            numerical_columns = [i for i in all_indices if i not in categorical_set]
            self.numerical_columns = sorted(numerical_columns)
            self.categorical_columns = sorted(categorical_columns)

    def _instantiate_encoder(self, settings: Dict[int, EncodingSettings]) -> Encoder:
        if self.in_loading or not self.use_encoder_cache:
            loaders = []
        else:
            if self.data is None:
                raise ValueError("`data` should be prepared when `in_loading` is False")
            train_loader, valid_loader = self.data.initialize()
            train_loader_copy = train_loader.copy()
            train_loader_copy.disable_shuffle()
            loaders = [train_loader_copy]
            if valid_loader is not None:
                assert isinstance(valid_loader, MLLoader)
                loaders.append(valid_loader)
        return Encoder(settings, config=self.encoder_config, loaders=loaders)

    def build_model(self, data_info: Dict[str, Any]) -> None:
        assert isinstance(self.input_dim, int)
        assert isinstance(self.output_dim, int)
        self.model = MLModel(
            self.output_dim,
            data_info["num_history"],
            encoder=self.encoder,
            use_encoder_cache=self.use_encoder_cache,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            use_one_hot=self.use_one_hot,
            use_embedding=self.use_embedding,
            only_categorical=self.only_categorical,
            core_name=self.core_name,
            core_config=self.core_config,
            pre_process_batch=self._pre_process_batch,
            num_repeat=self._num_repeat,
        )

    def build_inference(self) -> None:
        self.inference = MLInference(model=self.model)

    def prepare_trainer_defaults(self, data_info: Dict[str, Any]) -> None:  # type: ignore
        if self.trainer_config["monitor_names"] is None:
            self.trainer_config["monitor_names"] = ["mean_std", "plateau"]
            self._defaults["monitor_names"] = ["mean_std", "plateau"]
        super().prepare_trainer_defaults(data_info)
        if (
            self.trainer_config["metric_names"] is None
            and self.use_auto_loss
            and not self.trainer_config["use_losses_as_metrics"]
        ):
            if self.is_classification:
                self.trainer_config["metric_names"] = ["acc", "auc"]
            else:
                self.trainer_config["metric_names"] = ["mae", "mse"]
            self._defaults["metric_names"] = self.trainer_config["metric_names"]
        callback_names = self.trainer_config["callback_names"]
        if "_inject_loader_name" not in callback_names:
            callback_names.append("_inject_loader_name")
            cbs = self._defaults.setdefault("additional_callbacks", [])
            cbs.append("_inject_loader_name")

    ## api

    def build(
        self,
        data_info: Dict[str, Any],
        *,
        build_trainer: bool = True,
        report: Optional[bool] = None,
    ) -> None:
        super().build(data_info, build_trainer=build_trainer, report=report)
        self.processor.is_ready = True

    # load steps

    def permute_states(self, states: Dict[str, Any]) -> None:
        if self.encoder is not None:
            encoder_cache_keys = []
            for key in states:
                if key.startswith("encoder") and key.endswith("cache"):
                    encoder_cache_keys.append(key)
            for key in encoder_cache_keys:
                states.pop(key)

    # inference

    def postprocess(  # type: ignore
        self,
        outputs: InferenceOutputs,
        *,
        return_classes: bool = False,
        binary_threshold: float = 0.5,
        return_probabilities: bool = False,
    ) -> np_dict_type:
        forward = super().postprocess(outputs)
        if not self.is_classification:
            pass
        elif return_classes and return_probabilities:
            raise ValueError(
                "`return_classes` & `return_probabilities`"
                "should not be True at the same time"
            )
        elif not return_classes and not return_probabilities:
            pass
        else:
            predictions = forward[PREDICTIONS_KEY]
            if predictions.shape[1] > 2 and return_classes:
                forward[PREDICTIONS_KEY] = predictions.argmax(1, keepdims=True)
            else:
                probabilities = softmax(predictions)
                if return_probabilities:
                    forward[PREDICTIONS_KEY] = probabilities
                else:
                    assert probabilities.shape[1] == 2, "internal error occurred"
                    classes = (probabilities[..., [1]] >= binary_threshold).astype(int)
                    forward[PREDICTIONS_KEY] = classes
        # processor callback
        kw = dict(
            forward=forward,
            return_classes=return_classes,
            binary_threshold=binary_threshold,
            return_probabilities=return_probabilities,
        )
        forward = safe_execute(self.processor.postprocess_results, kw)
        return forward


@DLPipeline.register("ml")
class MLPipeline(IMLPipeline, DLPipeline, metaclass=ConfigMeta):  # type: ignore
    modifier = "ml"

    data: MLData
    model: MLModel
    inference: MLInference
    inference_base = MLInference

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
        use_encoder_cache: bool = False,
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_settings: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
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
        super().__init__(
            "MLModel",
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
            # when loading from json, the keys (idx) will be str
            self.encoding_settings[int(idx)] = EncodingSettings(**raw_setting)
        self._pre_process_batch = pre_process_batch
        self._num_repeat = num_repeat

    def make_inference_data(  # type: ignore
        self,
        x: data_type,
        y: data_type = None,
        *,
        shuffle: bool = False,
        sample_weights: sample_weights_type = None,
        **kwargs: Any,
    ) -> IMLData:
        kwargs["processor"] = self.data_info["processor"]
        kwargs["shuffle_train"] = shuffle
        kwargs["for_inference"] = True
        data: IMLData = IMLData.get(self.data_info["type"])(x, y, **kwargs)
        data.prepare(sample_weights)
        return data

    def to_pattern(
        self,
        *,
        binary_threshold: float = 0.5,
        pre_process: Optional[Callable[[IMLData], IMLData]] = None,
        **predict_kwargs: Any,
    ) -> ModelPattern:
        if ModelPattern is None:
            raise ValueError("`carefree-ml` need to be installed to use `to_pattern`")

        def _predict(idata: IMLData) -> np.ndarray:
            if pre_process is not None:
                idata = pre_process(idata)
            predictions = self.predict(idata, **predict_kwargs)[PREDICTIONS_KEY]
            if self.is_classification:
                return get_label_predictions(predictions, binary_threshold)
            return predictions

        def _predict_prob(idata: IMLData) -> np.ndarray:
            if not self.is_classification:
                msg = "`predict_prob` should not be called in regression tasks"
                raise ValueError(msg)
            if pre_process is not None:
                idata = pre_process(idata)
            logits = self.predict(idata, **predict_kwargs)[PREDICTIONS_KEY]
            logits = get_full_logits(logits)
            return softmax(logits)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    @staticmethod
    def fuse_multiple(
        export_folders: List[str],
        *,
        cuda: Optional[str] = None,
        compress: bool = True,
        build_trainer: bool = False,
        report: Optional[bool] = None,
        states_callback: states_callback_type = None,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[[DLPipeline, Dict[str, Any]], None]] = None,
    ) -> "MLPipeline":
        export_folder = export_folders[0]
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                m = IModifier.load_infrastructure(
                    MLPipeline,
                    export_folder,
                    cuda,
                    False,
                    pre_callback,
                    post_callback,
                )
                loaded = DataModule.load_info(export_folder, return_bytes=True)
                data_info = loaded.info
                m.data_module_bytes = loaded.data_module_bytes
        m._num_repeat = m.config["num_repeat"] = len(export_folders)
        m._make_modifier().build(data_info, build_trainer=build_trainer, report=report)
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


# carefree


class IMLCarefreeMakeModifier(Protocol):
    def __call__(self) -> "MLCarefreeModifier":
        pass


class IMLCarefreeMakeInferenceData(Protocol):
    def __call__(
        self,
        x: data_type,
        y: data_type = None,
        *,
        shuffle: bool = False,
        sample_weights: sample_weights_type = None,
        contains_labels: bool = True,
    ) -> MLCarefreeData:
        pass


class IMLCarefreePipeline:
    processor: _InternalCarefreeMLDataProcessor

    _make_modifier: IMLCarefreeMakeModifier
    make_inference_data: IMLCarefreeMakeInferenceData


_ml_carefree_requirements = get_requirements(IMLCarefreePipeline, excludes=[])


@IModifier.register("ml.carefree")
class MLCarefreeModifier(IMLCarefreePipeline, MLModifier):  # type: ignore
    requirements = MLModifier.requirements + _ml_carefree_requirements

    # build steps

    def setup_encoder(self) -> None:  # type: ignore
        excluded = 0
        encoding_settings = {}
        numerical_columns = []
        categorical_columns = []
        use_one_hot = False
        use_embedding = False
        cf_data = self.processor.cf_data
        if cf_data.is_simplify:
            numerical_columns = list(range(cf_data.processed.x.shape[1]))
        else:
            ts_indices = cf_data.ts_indices
            recognizers = cf_data.recognizers
            sorted_indices = [idx for idx in sorted(recognizers) if idx != -1]
            for idx in sorted_indices:
                recognizer = recognizers[idx]
                assert recognizer is not None
                if not recognizer.info.is_valid or idx in ts_indices:
                    excluded += 1
                elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                    numerical_columns.append(idx - excluded)
                elif idx not in encoding_settings:
                    true_idx = idx - excluded
                    setting = EncodingSettings(dim=recognizer.num_unique_values)
                    encoding_settings[true_idx] = setting
                    use_one_hot = use_one_hot or setting.use_one_hot
                    use_embedding = use_embedding or setting.use_embedding
                    categorical_columns.append(true_idx)
        if not encoding_settings:
            encoder = None
        else:
            encoder = self._instantiate_encoder(encoding_settings)
        self.encoder = encoder
        self.use_one_hot = use_one_hot
        self.use_embedding = use_embedding
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns


@DLPipeline.register("ml.carefree")
class MLCarefreePipeline(IMLCarefreePipeline, MLPipeline):  # type: ignore
    modifier = "ml.carefree"

    @property
    def cf_data(self) -> TabularData:
        return self.processor.cf_data


__all__ = [
    "MLModifier",
    "MLPipeline",
    "MLCarefreeModifier",
    "MLCarefreePipeline",
]
