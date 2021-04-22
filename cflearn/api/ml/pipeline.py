import os
import json
import shutil

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from functools import partial
from cfdata.tabular import ColumnTypes
from cfdata.tabular import TabularData
from cftool.ml import ModelPattern
from cftool.misc import lock_manager
from cftool.misc import Saving

from ...types import data_type
from ...types import np_dict_type
from ...types import states_callback_type
from ...trainer import get_sorted_checkpoints
from ...trainer import DeviceInfo
from ...protocol import loss_dict
from ...protocol import MetricProtocol
from ...protocol import InferenceOutputs
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import WARNING_PREFIX
from ...constants import PREDICTIONS_KEY
from ...constants import CHECKPOINTS_FOLDER
from ..internal_.pipeline import DLPipeline
from ...misc.toolkit import is_float
from ...misc.toolkit import get_arguments
from ...misc.toolkit import prepare_workplace_from
from ...misc.internal_ import MLData
from ...misc.internal_ import MLLoader
from ...misc.internal_ import MLInference
from ...models.ml.encoders import Encoder
from ...models.ml.protocol import MLModel


@DLPipeline.register("ml.simple")
class SimplePipeline(DLPipeline):
    model: MLModel
    inference: MLInference
    inference_base = MLInference

    train_loader: MLLoader
    valid_loader: Optional[MLLoader]

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
        output_dim: int,
        is_classification: bool,
        loss_name: str,
        loss_config: Optional[Dict[str, Any]] = None,
        # valid split
        valid_split: Optional[Union[int, float]] = None,
        min_valid_split: int = 100,
        max_valid_split: int = 10000,
        max_valid_split_ratio: float = 0.5,
        valid_split_order: str = "auto",
        # data loader
        num_history: int = 1,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
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
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        rank: Optional[int] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
        num_repeat: Optional[int] = None,
    ):
        self.config = get_arguments()
        self.config.pop("self")
        self.config.pop("__class__")
        super().__init__(
            loss_name=loss_name,
            loss_config=loss_config,
            valid_split=valid_split,
            min_valid_split=min_valid_split,
            max_valid_split=max_valid_split,
            max_valid_split_ratio=max_valid_split_ratio,
            valid_split_order=valid_split_order,
            num_history=num_history,
            shuffle_train=shuffle_train,
            shuffle_valid=shuffle_valid,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            state_config=state_config,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            valid_portion=valid_portion,
            amp=amp,
            clip_norm=clip_norm,
            metric_names=metric_names,
            metric_configs=metric_configs,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            optimizer_settings=optimizer_settings,
            workplace=workplace,
            rank=rank,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
        )
        self.core_name = core_name
        self.core_config = core_config or {}
        self.input_dim = None
        self.output_dim = output_dim
        self.is_classification = is_classification
        self.only_categorical = only_categorical
        self.encoder_config = encoder_config or {}
        self.encoding_methods = encoding_methods or {}
        self.encoding_configs = encoding_configs or {}
        if default_encoding_methods is None:
            default_encoding_methods = ["embedding"]
        self.default_encoding_methods = default_encoding_methods
        self.default_encoding_configs = default_encoding_configs or {}
        self._num_repeat: Optional[int] = num_repeat

    def _prepare_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_valid: Optional[np.ndarray],
        y_valid: Optional[np.ndarray],
    ) -> None:
        # prepare
        self.input_dim = x.shape[-1]
        train_loader = MLLoader(
            MLData(x, y),
            name="train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
        )
        if x_valid is None or y_valid is None:
            valid_loader = None
        else:
            valid_loader = MLLoader(
                MLData(x_valid, y_valid),
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
            )
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _prepare_data_attributes(self) -> None:
        self.encoder = None
        assert self.input_dim is not None
        self.numerical_columns_mapping = {i: i for i in range(self.input_dim)}
        self.categorical_columns_mapping = {}
        self.use_one_hot = False
        self.use_embedding = False

    def _prepare_modules(self) -> None:
        self._prepare_data_attributes()
        if not self.in_loading:
            workplace = prepare_workplace_from(self.trainer_config["workplace"])
            self.trainer_config["workplace"] = workplace
            self.trainer_config["metrics_log_file"] = self.metrics_log_file
            os.makedirs(workplace, exist_ok=True)
            with open(os.path.join(workplace, self.configs_file), "w") as f:
                json.dump(self.config, f)
        # prepare
        self.loss = loss_dict[self.loss_name](**(self.loss_config or {}))
        assert self.input_dim is not None
        self.model = MLModel(
            self.input_dim,
            self.output_dim,
            self.num_history,
            encoder=self.encoder,
            numerical_columns_mapping=self.numerical_columns_mapping,
            categorical_columns_mapping=self.categorical_columns_mapping,
            use_one_hot=self.use_one_hot,
            use_embedding=self.use_embedding,
            only_categorical=self.only_categorical,
            core_name=self.core_name,
            core_config=self.core_config,
            num_repeat=self._num_repeat,
        )
        self.inference = MLInference(model=self.model)

    def _prepare_trainer_defaults(self) -> None:
        # set some trainer defaults to ml tasks which work well in practice
        if self.trainer_config["monitor_names"] is None:
            self.trainer_config["monitor_names"] = ["mean_std", "plateau"]
        auto_callback_setup = False
        tqdm_settings = self.trainer_config["tqdm_settings"]
        callback_names = self.trainer_config["callback_names"]
        optimizer_settings = self.trainer_config["optimizer_settings"]
        if callback_names is None:
            callback_names = []
            auto_callback_setup = True
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        if "log_metrics_msg" not in callback_names and auto_callback_setup:
            if tqdm_settings is None or not tqdm_settings.get("use_tqdm", False):
                callback_names.append("log_metrics_msg")
        if "_default_opt_settings" not in callback_names:
            callback_names.append("_default_opt_settings")
        if "_inject_loader_name" not in callback_names:
            callback_names.append("_inject_loader_name")
        if optimizer_settings is None:
            optimizer_settings = {"all": {"optimizer": "adam", "scheduler": "warmup"}}
        self.trainer_config["tqdm_settings"] = tqdm_settings
        self.trainer_config["callback_names"] = callback_names
        self.trainer_config["optimizer_settings"] = optimizer_settings

    # TODO : support sample weights
    def _before_loop(
        self,
        x: data_type,
        y: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        cuda: Optional[str] = None,
    ) -> None:
        self._prepare_data(x, y, x_valid, y_valid)  # type: ignore
        self._prepare_modules()
        self._prepare_trainer_defaults()

    def _make_new_loader(  # type: ignore
        self,
        x: np.ndarray,
        batch_size: int,
        **kwargs: Any,
    ) -> MLLoader:
        return MLLoader(MLData(x, None), shuffle=False, batch_size=batch_size)

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
            return MetricProtocol.softmax(logits)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    @classmethod
    def pack(
        cls,
        workplace: str,
        *,
        input_dim: int,
        pack_name: str = "packed",
        cuda: Optional[str] = None,
    ) -> str:
        pack_folder = os.path.join(workplace, pack_name)
        if os.path.isdir(pack_folder):
            print(f"{WARNING_PREFIX}'{pack_folder}' already exists, it will be erased")
            shutil.rmtree(pack_folder)
        os.makedirs(pack_folder)
        abs_folder = os.path.abspath(pack_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [pack_folder]):
            checkpoint_folder = os.path.join(workplace, CHECKPOINTS_FOLDER)
            scores_path = os.path.join(checkpoint_folder, SCORES_FILE)
            best_file = get_sorted_checkpoints(scores_path)[0]
            new_file = f"{PT_PREFIX}-1.pt"
            shutil.copy(
                os.path.join(checkpoint_folder, best_file),
                os.path.join(pack_folder, new_file),
            )
            with open(scores_path, "r") as rf:
                scores = json.load(rf)
            with open(os.path.join(pack_folder, SCORES_FILE), "w") as wf:
                json.dump({new_file: scores[best_file]}, wf)
            with open(os.path.join(workplace, cls.configs_file), "r") as rf:
                config = json.load(rf)
            config_bundle = {
                "config": config,
                "device_info": DeviceInfo(cuda, None),
                "input_dim": input_dim,
            }
            Saving.save_dict(config_bundle, cls.config_bundle_name, pack_folder)
            Saving.compress(abs_folder, remove_original=True)
        return pack_folder

    def re_save(self, export_folder: str) -> "SimplePipeline":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            self._save_misc(export_folder, False)
            file = f"{PT_PREFIX}-1.pt"
            torch.save(self.model.state_dict(), os.path.join(export_folder, file))
            Saving.compress(abs_folder, remove_original=True)
        return self


@DLPipeline.register("ml.carefree")
class CarefreePipeline(SimplePipeline):
    data: TabularData
    train_data: TabularData
    valid_data: Optional[TabularData]

    def __init__(
        self,
        core_name: str = "fcnn",
        core_config: Optional[Dict[str, Any]] = None,
        *,
        output_dim: Optional[int] = None,
        is_classification: Optional[bool] = None,
        loss_name: str = "auto",
        loss_config: Optional[Dict[str, Any]] = None,
        # data
        data_config: Optional[Dict[str, Any]] = None,
        read_config: Optional[Dict[str, Any]] = None,
        # valid split
        valid_split: Optional[Union[int, float]] = None,
        min_valid_split: int = 100,
        max_valid_split: int = 10000,
        max_valid_split_ratio: float = 0.5,
        valid_split_order: str = "auto",
        # data loader
        num_history: int = 1,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
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
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        rank: Optional[int] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
    ):
        self.config = get_arguments()
        self.config.pop("self")
        self.config.pop("__class__")
        super().__init__(
            core_name,
            core_config,
            output_dim=output_dim,  # type: ignore
            is_classification=is_classification,  # type: ignore
            loss_name=loss_name,
            loss_config=loss_config,
            valid_split=valid_split,
            min_valid_split=min_valid_split,
            max_valid_split=max_valid_split,
            max_valid_split_ratio=max_valid_split_ratio,
            valid_split_order=valid_split_order,
            num_history=num_history,
            shuffle_train=shuffle_train,
            shuffle_valid=shuffle_valid,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            only_categorical=only_categorical,
            encoder_config=encoder_config,
            encoding_methods=encoding_methods,
            encoding_configs=encoding_configs,
            default_encoding_methods=default_encoding_methods,
            default_encoding_configs=default_encoding_configs,
            state_config=state_config,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            valid_portion=valid_portion,
            amp=amp,
            clip_norm=clip_norm,
            metric_names=metric_names,
            metric_configs=metric_configs,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            optimizer_settings=optimizer_settings,
            workplace=workplace,
            rank=rank,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
        )
        self.core_name = core_name
        self.core_config = core_config or {}
        if data_config is None:
            data_config = {}
        data_config["default_categorical_process"] = "identical"
        if is_classification is not None:
            data_config["task_type"] = "clf" if is_classification else "reg"
        self.data = TabularData(**(data_config or {}))
        self.read_config = read_config or {}

    # TODO : support sample weights
    def _prepare_data(
        self,
        x: data_type,
        y: data_type,
        x_valid: data_type,
        y_valid: data_type,
    ) -> None:
        self.data.read(x, y, **self.read_config)
        if x_valid is not None:
            self.train_data = self.data
            self.valid_data = self.data.copy_to(x_valid, y_valid)
        else:
            if isinstance(self.valid_split, int):
                split = self.valid_split
            else:
                num_data = len(self.data)
                if isinstance(self.valid_split, float):
                    split = int(round(self.valid_split * num_data))
                else:
                    default_split = 0.1
                    num_split = int(round(default_split * num_data))
                    num_split = max(self.min_valid_split, num_split)
                    max_split = int(round(num_data * self.max_valid_split_ratio))
                    max_split = min(max_split, self.max_valid_split)
                    split = min(num_split, max_split)
            if split <= 0:
                self.train_data = self.data
                self.valid_data = None
            else:
                split_result = self.data.split(split, order=self.valid_split_order)
                self.train_data = split_result.remained
                self.valid_data = split_result.split
        train_loader = MLLoader(
            MLData(*self.train_data.processed.xy),
            name="train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
        )
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = MLLoader(
                MLData(*self.valid_data.processed.xy),
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
            )
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _prepare_data_attributes(self) -> None:
        self.is_classification = self.data.is_clf
        if self.input_dim is None:
            self.input_dim = self.data.processed_dim
        if self.output_dim is None:
            self.output_dim = 1 if self.data.is_reg else self.data.num_classes
        if self.loss_name == "auto":
            self.loss_name = "focal" if self.data.is_clf else "mae"

    def _prepare_modules(self) -> None:
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
        if self.data.is_simplify:
            for idx in range(self.data.processed.x.shape[1]):
                numerical_columns_mapping[idx] = idx
        else:
            ts_indices = self.data.ts_indices
            recognizers = self.data.recognizers
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
                loaders = [self.train_loader.copy()]
                if self.valid_loader is not None:
                    loaders.append(self.valid_loader)
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
        super()._prepare_modules()

    def _prepare_trainer_defaults(self) -> None:
        if self.trainer_config["metric_names"] is None:
            if self.data.is_clf:
                self.trainer_config["metric_names"] = ["acc", "auc"]
            else:
                self.trainer_config["metric_names"] = ["mae", "mse"]
        super()._prepare_trainer_defaults()

    def _predict_from_outputs(self, outputs: InferenceOutputs) -> np_dict_type:
        results = outputs.forward_results
        if self.data.is_clf:
            return results
        fn = partial(self.data.recover_labels, inplace=True)
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
        x: data_type,
        batch_size: int,
        **kwargs: Any,
    ) -> MLLoader:
        return MLLoader(
            MLData(*self.data.transform(x, None, **kwargs).xy),
            shuffle=False,
            batch_size=batch_size,
        )

    def _save_misc(self, export_folder: str, retain_data: bool) -> float:
        data_folder = os.path.join(export_folder, self.data_folder)
        self.data.save(data_folder, retain_data=retain_data, compress=False)
        return super()._save_misc(export_folder, retain_data)

    @classmethod
    def _load_infrastructure(
        cls,
        export_folder: str,
        cuda: Optional[str],
    ) -> "CarefreePipeline":
        m = super()._load_infrastructure(export_folder, cuda)
        assert isinstance(m, CarefreePipeline)
        data_folder = os.path.join(export_folder, cls.data_folder)
        m.data = TabularData.load(data_folder, compress=False)
        return m

    @classmethod
    def load(
        cls,
        export_folder: str,
        *,
        compress: bool = True,
        states_callback: states_callback_type = None,
    ) -> "CarefreePipeline":
        def _callback(m_: Any, states_: Dict[str, Any]) -> Dict[str, Any]:
            if states_callback is not None:
                states_ = states_callback(m_, states_)
            if m_.encoder is not None:
                encoder_cache_keys = []
                for key in states_:
                    if key.startswith("encoder") and key.endswith("cache"):
                        encoder_cache_keys.append(key)
                for key in encoder_cache_keys:
                    states_.pop(key)
            return states_

        m = super().load(export_folder, compress=compress, states_callback=_callback)
        assert isinstance(m, CarefreePipeline)
        return m

    @classmethod
    def pack(
        cls,
        workplace: str,
        *,
        input_dim: int,
        pack_name: str = "packed",
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
