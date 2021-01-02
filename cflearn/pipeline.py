import os
import json
import torch
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from typing import *
from tqdm import tqdm
from cfdata.tabular import TabularDataset
from cftool.ml import ModelPattern
from cftool.misc import show_or_save
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import timing_context
from cftool.misc import Saving
from cftool.misc import LoggingMixin

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from .types import data_type
from .configs import Elements
from .configs import Environment
from .trainer import Trainer
from .trainer import IntermediateResults
from .protocol import DataProtocol
from .protocol import PrefetchLoader
from .protocol import DataLoaderProtocol
from .inference import Inference
from .inference import PreProcessor
from .misc._api import _fetch_saving_paths
from .misc.toolkit import to_2d
from .misc.toolkit import to_relative
from .misc.toolkit import eval_context
from .misc.toolkit import LoggingMixinWithRank
from .misc.time_series import TSLabelCollator
from .models.base import model_dict
from .models.base import ModelBase
from .models.base import PipeConfig


key_type = Tuple[Union[str, Optional[str]], ...]


class Pipeline(LoggingMixinWithRank):
    config_bundle_name = "config_bundle"

    def __init__(self, environment: Environment):
        # typing
        self.tr_data: DataProtocol
        self.cv_data: Optional[DataProtocol]
        self.tr_loader: DataLoaderProtocol
        self.tr_loader_copy: DataLoaderProtocol
        self.cv_loader: Optional[DataLoaderProtocol]
        # common
        self.environment = environment
        self.device = environment.device
        self.model: Optional[ModelBase] = None
        self.inference: Optional[Inference]
        LoggingMixin.reset_logging()
        self.config = environment.pipeline_config
        self.model_type = environment.model
        self.timing = self.config.setdefault("use_timing_context", True)
        self.data_config["use_timing_context"] = self.timing
        self.data_config["default_categorical_process"] = "identical"
        self.sampler_config = self.config.setdefault("sampler_config", {})

    def __getattr__(self, item: str) -> Any:
        return self.environment.config.get(item)

    def __str__(self) -> str:
        return f"{type(self.model).__name__}()"  # type: ignore

    __repr__ = __str__

    @property
    def data(self) -> DataProtocol:
        return self._original_data

    @property
    def train_set(self) -> TabularDataset:
        raw = self.tr_data.raw
        return TabularDataset(*raw.xy, task_type=self.tr_data.task_type)

    @property
    def valid_set(self) -> Optional[TabularDataset]:
        if self.cv_data is None:
            return None
        raw = self.cv_data.raw
        return TabularDataset(*raw.xy, task_type=self.cv_data.task_type)

    @property
    def int_cv_split(self) -> int:
        if isinstance(self.cv_split, int):
            return self.cv_split
        num_data = len(self._original_data)
        if self.cv_split is not None:
            return int(round(self.cv_split * num_data))
        default_cv_split = 0.1
        cv_split_num = int(round(default_cv_split * num_data))
        cv_split_num = max(self.min_cv_split, cv_split_num)
        max_cv_split = int(round(num_data * self.max_cv_split_ratio))
        max_cv_split = min(self.max_cv_split, max_cv_split)
        return min(cv_split_num, max_cv_split)

    @property
    def binary_threshold(self) -> Optional[float]:
        if self.inference is None:
            raise ValueError("`inference` is not yet generated")
        return self.inference.binary_threshold

    @property
    def user_config(self) -> Dict[str, Any]:
        return shallow_copy_dict(self.environment.user_config)

    @property
    def user_inc_config(self) -> Dict[str, Any]:
        return shallow_copy_dict(self.environment.user_increment_config)

    def _init_data(self) -> None:
        if not self.data.is_ts:
            self.ts_label_collator = None
        else:
            self.ts_label_collator = TSLabelCollator(
                self.data,
                self.ts_label_collator_config,
            )
        self.sampler_config.setdefault("verbose_level", self.data._verbose_level)
        self.preprocessor = PreProcessor(
            self._original_data,
            self.loader_protocol,
            self.sampler_protocol,
            self.sampler_config,
        )
        tr_sampler = self.preprocessor.make_sampler(
            self.tr_data,
            self.shuffle_tr,
            self.tr_weights,
        )
        self.tr_loader = DataLoaderProtocol.make(
            self.loader_protocol,
            self.batch_size,
            tr_sampler,
            return_indices=True,
            verbose_level=self._verbose_level,
            label_collator=self.ts_label_collator,
        )
        if self.cv_data is None:
            self.cv_loader = None
        else:
            cv_sampler = self.preprocessor.make_sampler(self.cv_data, False)
            self.cv_loader = DataLoaderProtocol.make(
                self.loader_protocol,
                self.cv_batch_size,
                cv_sampler,
                return_indices=True,
                verbose_level=self._verbose_level,
                label_collator=self.ts_label_collator,
            )
            self.cv_loader.enabled_sampling = False
        # tr loader copy
        self.tr_loader_copy = self.tr_loader.copy()
        self.tr_loader_copy.enabled_sampling = False
        self.tr_loader_copy.sampler.shuffle = False

    def _prepare_modules(
        self,
        *,
        is_loading: bool = False,
        loaded_registered_pipes: Optional[Dict[str, PipeConfig]] = None,
    ) -> None:
        # logging
        if not is_loading:
            if os.path.isdir(self.logging_folder):
                if os.listdir(self.logging_folder):
                    print(
                        f"{self.warning_prefix}'{self.logging_folder}' already exists, "
                        "it will be cleared up to store our logging"
                    )
                shutil.rmtree(self.logging_folder)
            os.makedirs(self.logging_folder)
        self._init_logging(self.verbose_level, self.trigger_logging)
        # model
        with timing_context(self, "init model", enable=self.timing):
            self.model = model_dict[self.model_type](
                self.environment,
                self.tr_loader_copy,
                self.cv_loader,
                self.tr_weights,
                self.cv_weights,
                loaded_registered_pipes,
            )
            self.model.init_ema()
        # trainer
        with timing_context(self, "init trainer", enable=self.timing):
            if self.preprocessor is None:
                msg = "`preprocessor` is not defined. Please call `_init_data` first"
                raise ValueError(msg)
            self.inference = Inference(
                self.preprocessor,
                model=self.model,
                binary_config=self.binary_config,
                use_binary_threshold=self.use_binary_threshold,
                use_tqdm=self.use_tqdm,
            )
            self.trainer = Trainer(
                self.model,
                self.inference,
                self.environment,
                is_loading,
            )
        # to device
        with timing_context(self, "init device", enable=self.timing):
            self.model.to(self.device)

    def _before_loop(
        self,
        x: data_type,
        y: data_type,
        x_cv: data_type,
        y_cv: data_type,
        sample_weights: np.ndarray,
    ) -> None:
        # data
        y, y_cv = map(to_2d, [y, y_cv])
        args = (x, y) if y is not None else (x,)
        self.data_config["verbose_level"] = self._verbose_level
        if sample_weights is None:
            self.sample_weights = None
        else:
            self.sample_weights = sample_weights.copy()
        self._original_data = DataProtocol.make(self.data_protocol, **self.data_config)
        self._original_data.read(*args, **self.read_config)
        self.tr_data = self._original_data
        self._save_original_data = x_cv is None
        self.tr_weights = self.cv_weights = None
        if x_cv is not None:
            self.cv_data = self.tr_data.copy_to(x_cv, y_cv)
            if sample_weights is not None:
                self.tr_weights = sample_weights[: len(self.tr_data)]
                self.cv_weights = sample_weights[len(self.tr_data) :]
        else:
            if self.int_cv_split <= 0:
                self.cv_data = None
                self.tr_split_indices = None
                self.cv_split_indices = None
                if sample_weights is not None:
                    self.tr_weights = sample_weights
            else:
                split = self.tr_data.split(self.int_cv_split, order=self.cv_split_order)
                self.tr_data, self.cv_data = split.remained, split.split
                self.tr_split_indices = split.remained_indices
                self.cv_split_indices = split.split_indices
                # TODO : utilize cv_weights with sample_weights[split.split_indices]
                if sample_weights is not None:
                    self.tr_weights = sample_weights[split.remained_indices]
                    self.cv_weights = sample_weights[split.split_indices]
        # deep speed
        self.set_rank_0(self.environment.is_rank_0)
        # data
        self._init_data()
        # modules
        self._prepare_modules()
        # deep speed
        self.set_rank_0(self.is_rank_0)

    def _handle_pretrain(
        self,
        strict: bool,
        folder: Optional[str],
        identifier: Optional[str],
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if identifier is None:
            return None
        paths_dict = _fetch_saving_paths(identifier, folder)
        all_paths: List[str] = sum(paths_dict.values(), [])
        if len(all_paths) > 1:
            raise ValueError("more than 1 model is detected")
        path = all_paths[0]
        folder = folder or "./"
        compress = os.path.isfile(f"{path}.zip")
        with lock_manager(folder, [path]):
            with Saving.compress_loader(path, compress):
                self.trainer.restore_checkpoint(path, strict, state_dict_callback)

    def _loop(self) -> None:
        # dump information
        logging_folder = self.logging_folder
        os.makedirs(logging_folder, exist_ok=True)
        if self.is_rank_0:
            if self.environment.deepspeed:
                logging_folder = os.path.join(logging_folder, os.pardir)
            Saving.save_dict(self.config, "__config__", logging_folder)
            with open(os.path.join(logging_folder, "__model__.txt"), "w") as f:
                f.write(str(self.model))
        # training loop
        self.trainer.fit(
            self.tr_loader,
            self.tr_loader_copy,
            self.cv_loader,
            self.tr_weights,
            self.cv_weights,
        )
        self.log_timing()

    @staticmethod
    def _rectangle(
        ax: Any,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Any,
        text: str,
    ) -> Tuple[float, float]:
        rectangle = mpatch.Rectangle(
            (x, y),
            width,
            height,
            color=color,
            alpha=0.8,
            ec="#000000",
        )
        ax.add_artist(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + 0.5 * rectangle.get_width()
        cy = ry + 0.5 * rectangle.get_height()
        ax.annotate(
            text,
            (cx, cy),
            color="black",
            fontsize=16,
            ha="center",
            va="center",
        )
        return cx, cy

    @staticmethod
    def _arrow(
        ax: Any,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        half_box_width: float,
    ) -> None:
        lx += half_box_width
        rx -= half_box_width
        ax.annotate(
            text="",
            xy=(rx, ry),
            xytext=(lx, ly),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    @staticmethod
    def _box_msg(key: key_type, delim: str) -> str:
        scope, meta, config = key[1:]
        if meta is None:
            scope_str = ""
            extractor_str = f"{scope}_{config}"
        else:
            scope_str = f"\n{scope}"
            extractor_str = f"{meta}_{config}"
        return f"Extractor\n{delim}\n{extractor_str}{scope_str}"

    @staticmethod
    def _rectangles(
        ax: Any,
        color: Any,
        x: float,
        y_max: float,
        box_width: float,
        box_height: float,
        delim: str,
        keys: List[key_type],
        positions: Dict[key_type, Tuple[float, float]],
    ) -> None:
        for i, key in enumerate(keys):
            y = (i + 0.5) * (y_max / len(keys))
            args = ax, x, y, box_width, box_height, color
            cx, cy = Pipeline._rectangle(*args, Pipeline._box_msg(key, delim))
            positions[key] = cx, cy

    # api

    def draw(
        self,
        export_path: Optional[str] = None,
        *,
        transparent: bool = True,
    ) -> "Pipeline":
        pipes = model_dict[self.model_type].registered_pipes
        if pipes is None:
            raise ValueError("pipes have not yet been registered")
        transforms_mapping: Dict[str, str] = {}
        extractors_mapping: Dict[str, key_type] = {}
        heads_mapping: Dict[str, key_type] = {}
        sorted_keys = sorted(pipes)
        for key in sorted_keys:
            pipe_cfg = pipes[key]
            transforms_mapping[key] = pipe_cfg.transform
            extractor_key: key_type = (
                pipe_cfg.transform,
                pipe_cfg.extractor,
                pipe_cfg.extractor_meta_scope,
                pipe_cfg.extractor_config,
            )
            if not pipe_cfg.reuse_extractor:
                cursor = 0
                new_extractor_key: key_type = extractor_key
                while new_extractor_key in extractors_mapping.values():
                    cursor += 1
                    new_extractor_key = extractor_key + (str(cursor),)
                extractor_key = new_extractor_key
            extractors_mapping[key] = extractor_key
            heads_mapping[key] = (
                key,
                pipe_cfg.head,
                pipe_cfg.head_meta_scope,
                pipe_cfg.head_config,
            )
        unique_transforms = sorted(set(transforms_mapping.values()))
        unique_extractors = sorted(set(extractors_mapping.values()))
        all_heads = sorted(heads_mapping.values())

        box_width = 0.5
        box_height = 0.4
        half_box_width = 0.5 * box_width
        x_scale, y_scale = 6, 5
        x_positions = [0, 0.75, 1.5, 2.25]
        y_gap = box_height * 2.5
        x_min, x_max = x_positions[0], x_positions[-1]
        x_diff = x_max - x_min
        nodes_list = [unique_transforms, unique_extractors, all_heads]
        y_max = float(max(map(len, nodes_list))) * y_gap  # type: ignore
        y_max = max(1.5, y_max)
        fig = plt.figure(dpi=100, figsize=[(x_diff + 2.0) * x_scale, y_max * y_scale])

        ax = fig.add_subplot(111)
        if transparent:
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.tick_params(labelleft=False, left=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        colors = plt.cm.Paired([i / 6 for i in range(4)])

        # rectangle
        delim = "-" * 16
        color = colors[0]
        x = x_positions[0]
        transform_positions = {}
        extractor_positions: Dict[key_type, Tuple[float, float]] = {}
        head_positions: Dict[key_type, Tuple[float, float]] = {}
        for i, transform in enumerate(unique_transforms):
            y = (i + 0.5) * (y_max / len(unique_transforms))
            args = ax, x, y, box_width, box_height, color
            cx, cy = self._rectangle(*args, f"Transform\n{delim}\n{transform}")
            transform_positions[transform] = cx, cy
        color = colors[1]
        x = x_positions[1]
        self._rectangles(
            ax,
            color,
            x,
            y_max,
            box_width,
            box_height,
            delim,
            unique_extractors,
            extractor_positions,
        )
        color = colors[2]
        x = x_positions[2]
        self._rectangles(
            ax,
            color,
            x,
            y_max,
            box_width,
            box_height,
            delim,
            all_heads,
            head_positions,
        )
        color = colors[3]
        x, y = x_positions[3], 0.5 * y_max
        args = ax, x, y, box_width, box_height, color
        aggregator = self.environment.model_config["aggregator"]
        cx, cy = self._rectangle(*args, f"Aggregator\n{delim}\n{aggregator}")
        aggregator_position = cx, cy

        # arrows
        for key in sorted_keys:
            transform = transforms_mapping[key]
            extractor = extractors_mapping[key]
            head_tuple = heads_mapping[key]
            x1, y1 = transform_positions[transform]
            x2, y2 = extractor_positions[extractor]
            x3, y3 = head_positions[head_tuple]
            self._arrow(ax, x1, y1, x2, y2, half_box_width)
            self._arrow(ax, x2, y2, x3, y3, half_box_width)
            self._arrow(ax, x3, y3, *aggregator_position, half_box_width)

        ax.set_xlim(x_min, x_max + box_width + 0.1)
        ax.set_ylim(0, y_max + box_height)
        show_or_save(export_path, fig)
        return self

    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        pretrain_strict: bool = False,
        pretrain_folder: Optional[str] = None,
        pretrain_identifier: Optional[str] = None,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "Pipeline":
        self._before_loop(x, y, x_cv, y_cv, sample_weights)
        self._handle_pretrain(
            pretrain_strict,
            pretrain_folder,
            pretrain_identifier,
            state_dict_callback,
        )
        self._loop()
        # finalize mlflow
        run_id = self.trainer.run_id
        mlflow_client = self.trainer.mlflow_client
        if mlflow_client is not None:
            # log model
            if self.production is not None:
                import mlflow
                import cflearn

                cwd = os.getcwd()
                root_folder = os.path.join(os.path.dirname(__file__), os.pardir)
                conda_env = os.path.join(os.path.abspath(root_folder), "conda.yml")
                if self.production == "pack":
                    pack_folder = os.path.join(self.logging_folder, "__packed__")
                    pack_folder = to_relative(os.path.abspath(pack_folder), cwd)
                    cflearn.Pack.pack(self, pack_folder, compress=False, verbose=False)
                    mlflow.pyfunc.save_model(
                        os.path.join(self.logging_folder, "__pyfunc__"),
                        python_model=cflearn.PackModel(),
                        artifacts={"pack_folder": pack_folder},
                        conda_env=conda_env,
                    )
                    cflearn._rmtree(pack_folder)
                elif self.production == "pipeline":
                    export_folder = os.path.join(self.logging_folder, "pipeline")
                    export_folder = to_relative(os.path.abspath(export_folder), cwd)
                    self.save(export_folder, compress=False)
                    mlflow.pyfunc.save_model(
                        os.path.join(self.logging_folder, "__pyfunc__"),
                        python_model=cflearn.PipelineModel(),
                        artifacts={"export_folder": export_folder},
                        conda_env=conda_env,
                    )
                else:
                    msg = f"unrecognized production type '{self.production}' found"
                    raise NotImplementedError(msg)
            # log artifacts
            if self.environment.log_pipeline_to_artifacts:
                if self.production != "pipeline":
                    self.save(os.path.join(self.logging_folder, "pipeline"))
            self.trainer._log_artifacts()
            # terminate
            mlflow_client.set_terminated(run_id)
        return self

    def predict(
        self,
        x: data_type,
        *,
        return_all: bool = False,
        contains_labels: bool = False,
        requires_recover: bool = True,
        returns_probabilities: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.inference is None:
            raise ValueError("`inference` is not yet generated")
        loader = self.preprocessor.make_inference_loader(
            x,
            self.device,
            self.cv_batch_size,
            is_onnx=self.inference.onnx is not None,
            contains_labels=contains_labels,
        )
        kwargs = shallow_copy_dict(kwargs)
        kwargs.update(
            {
                "return_all": return_all,
                "requires_recover": requires_recover,
                "returns_probabilities": returns_probabilities,
            }
        )

        if self.inference is None:
            raise ValueError("`inference` is not yet generated")
        return self.inference.predict(loader, **shallow_copy_dict(kwargs))

    def predict_prob(
        self,
        x: data_type,
        *,
        return_all: bool = False,
        contains_labels: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.data.is_reg:
            raise ValueError("`predict_prob` should not be called on regression tasks")
        return self.predict(
            x,
            return_all=return_all,
            contains_labels=contains_labels,
            returns_probabilities=True,
            **shallow_copy_dict(kwargs),
        )

    def to_pattern(
        self,
        *,
        pre_process: Optional[Callable] = None,
        **predict_kwargs: Any,
    ) -> ModelPattern:
        def _predict(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            return self.predict(x, **predict_kwargs)

        def _predict_prob(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            return self.predict_prob(x, **predict_kwargs)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    data_folder = "data"
    train_folder = "train"
    valid_folder = "valid"
    original_folder = "original"
    train_indices_file = "train_indices.npy"
    valid_indices_file = "valid_indices.npy"
    sample_weights_file = "sample_weights.npy"
    final_results_file = "final_results.json"
    registered_pipes_file = "registered_pipes.json"

    @classmethod
    def make(
        cls,
        config: Dict[str, Any],
        increment_config: Dict[str, Any],
    ) -> "Pipeline":
        return cls(Environment.from_elements(Elements.make(config, increment_config)))

    def save(
        self,
        export_folder: Optional[str] = None,
        *,
        compress: bool = True,
        remove_original: bool = True,
    ) -> "Pipeline":
        if export_folder is None:
            export_folder = self.trainer.checkpoint_folder
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            # data
            data_folder = os.path.join(export_folder, self.data_folder)
            os.makedirs(data_folder, exist_ok=True)
            if self.sample_weights is not None:
                sw_file = os.path.join(data_folder, self.sample_weights_file)
                np.save(sw_file, self.sample_weights)
            if not self._save_original_data:
                assert self.cv_data is not None
                train_data_folder = os.path.join(data_folder, self.train_folder)
                valid_data_folder = os.path.join(data_folder, self.valid_folder)
                self.tr_data.save(train_data_folder, compress=False)
                self.cv_data.save(valid_data_folder, compress=False)
            else:
                original_data_folder = os.path.join(data_folder, self.original_folder)
                self._original_data.save(original_data_folder, compress=False)
                if self.tr_split_indices is not None:
                    tr_file = os.path.join(data_folder, self.train_indices_file)
                    np.save(tr_file, self.tr_split_indices)
                if self.cv_split_indices is not None:
                    cv_file = os.path.join(data_folder, self.valid_indices_file)
                    np.save(cv_file, self.cv_split_indices)
            # registered pipes
            if self.model is None:
                raise ValueError("`model` is not yet generated")
            pipes = self.model.registered_pipes
            pipes_path = os.path.join(export_folder, self.registered_pipes_file)
            with open(pipes_path, "w") as f:
                json.dump(pipes, f)
            # final results
            final_results = self.trainer.final_results
            if final_results is None:
                raise ValueError("`final_results` are not generated yet")
            with open(os.path.join(export_folder, self.final_results_file), "w") as f:
                json.dump(final_results, f)
            # pytorch checkpoint
            score = final_results.final_score
            self.trainer.save_checkpoint(score, export_folder)
            # misc config bundle
            if self.inference is None:
                raise ValueError("`inference` is not yet generated")
            config_bundle = {
                "config": shallow_copy_dict(self.user_config),
                "increment_config": shallow_copy_dict(self.user_inc_config),
                "binary_config": self.inference.binary_config,
            }
            Saving.save_dict(config_bundle, self.config_bundle_name, export_folder)
            # compress
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def load(cls, export_folder: str, *, compress: bool = True) -> "Pipeline":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                # misc config bundle
                config_bundle = Saving.load_dict(cls.config_bundle_name, export_folder)
                user_config = config_bundle["config"]
                user_increment_config = config_bundle["increment_config"]
                user_increment_config["binary_config"] = config_bundle["binary_config"]
                user_increment_config["verbose_level"] = 0
                pipeline = cls.make(user_config, user_increment_config)
                # sample weights
                data_folder = os.path.join(export_folder, cls.data_folder)
                tr_weights = cv_weights = sample_weights = None
                sw_file = os.path.join(data_folder, cls.sample_weights_file)
                if os.path.isfile(sw_file):
                    sample_weights = np.load(sw_file)
                # data
                cv_data: Optional[DataProtocol]
                data_base = DataProtocol.get(pipeline.data_protocol)
                original_data_folder = os.path.join(data_folder, cls.original_folder)
                if not os.path.isdir(original_data_folder):
                    train_data_folder = os.path.join(data_folder, cls.train_folder)
                    valid_data_folder = os.path.join(data_folder, cls.valid_folder)
                    try:
                        tr_data = data_base.load(train_data_folder, compress=False)
                        cv_data = data_base.load(valid_data_folder, compress=False)
                    except Exception as e:
                        raise ValueError(
                            f"data information is corrupted ({e}), "
                            "this may cause by backward compatible breaking"
                        )
                    original_data = tr_data
                    if sample_weights is not None:
                        tr_weights = sample_weights[: len(tr_data)]
                        cv_weights = sample_weights[len(tr_data) :]
                else:
                    original_data = data_base.load(
                        original_data_folder,
                        compress=False,
                    )
                    vi_file = os.path.join(data_folder, cls.valid_indices_file)
                    if not os.path.isfile(vi_file):
                        tr_weights = sample_weights
                        tr_data = original_data
                        cv_data = None
                    else:
                        ti_file = os.path.join(data_folder, cls.train_indices_file)
                        train_indices, valid_indices = map(np.load, [ti_file, vi_file])
                        split = original_data.split_with_indices(
                            valid_indices, train_indices
                        )
                        tr_data, cv_data = split.remained, split.split
                        if sample_weights is not None:
                            tr_weights = sample_weights[train_indices]
                            cv_weights = sample_weights[valid_indices]
                pipeline.sample_weights = sample_weights
                pipeline.tr_weights = tr_weights
                pipeline.cv_weights = cv_weights
                pipeline._original_data = original_data
                pipeline.tr_data = tr_data
                pipeline.cv_data = cv_data
                pipeline._init_data()
                # registered pipes
                pipes_path = os.path.join(export_folder, cls.registered_pipes_file)
                if not os.path.isfile(pipes_path):
                    pipes = None
                else:
                    with open(pipes_path, "r") as f:
                        pipes = {k: PipeConfig(*v) for k, v in json.load(f).items()}
                # prepare modules
                pipeline._prepare_modules(
                    is_loading=True,
                    loaded_registered_pipes=pipes,
                )
                trainer = pipeline.trainer
                trainer.state.inject_loader(pipeline.tr_loader)
                trainer.tr_loader = PrefetchLoader(pipeline.tr_loader, pipeline.device)
                cv_loader = pipeline.cv_loader
                if cv_loader is None:
                    trainer.cv_loader = None
                else:
                    trainer.cv_loader = PrefetchLoader(cv_loader, pipeline.device)
                # pytorch checkpoint
                trainer.restore_checkpoint(export_folder)
                # final results
                trainer._init_metrics()
                final_results_path = os.path.join(export_folder, cls.final_results_file)
                with open(final_results_path, "r") as f:
                    trainer.final_results = IntermediateResults(*json.load(f))
        return pipeline

    def profile_forward(self, *, num_repeat: int = 100, **kwargs: Any) -> None:
        if self.model is None:
            raise ValueError("`model` is not generated yet")
        input_sample = {k: v.to("cpu") for k, v in self.model.input_sample.items()}
        self.model.to("cpu")
        with eval_context(self.model):
            with torch.autograd.profiler.profile(**kwargs) as prof:
                for _ in tqdm(range(num_repeat), total=num_repeat):
                    self.model(input_sample)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        self.model.to(self.device)


__all__ = ["Pipeline"]
