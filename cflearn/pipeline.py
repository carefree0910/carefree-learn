import os
import json
import math
import torch
import shutil
import inspect
import zipfile

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
from cftool.misc import random_hash
from cftool.misc import safe_execute
from cftool.misc import print_warning
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import prepare_workplace_from
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import WithRegister
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .data import DataModule
from .data import DLDataModule
from .types import configs_type
from .types import sample_weights_type
from .types import states_callback_type
from .trainer import get_sorted_checkpoints
from .trainer import Trainer
from .protocol import loss_dict
from .protocol import callback_dict
from .protocol import DeviceInfo
from .protocol import ILoss
from .protocol import IDLModel
from .protocol import _IMetric
from .protocol import MetricsOutputs
from .protocol import InferenceOutputs
from .protocol import IInference
from .protocol import IDataLoader
from .protocol import ModelWithCustomSteps
from .constants import PT_PREFIX
from .constants import SCORES_FILE
from .constants import CHECKPOINTS_FOLDER
from .constants import BATCH_INDICES_KEY
from .misc.toolkit import get_ddp_info
from .misc.toolkit import _get_environ_workplace
from .misc.toolkit import ConfigMeta
from .misc.internal_.trainer import make_trainer


pipeline_dict: Dict[str, Type["IPipeline"]] = {}
dl_pipeline_modifiers: Dict[str, Type["IModifier"]] = {}


class IPipeline(WithRegister["IPipeline"], metaclass=ABCMeta):
    d = pipeline_dict

    @abstractmethod
    def build(self, data_info: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        data: DataModule,
        *,
        sample_weights: sample_weights_type = None,
    ) -> "IPipeline":
        pass

    @abstractmethod
    def predict(self, data: DataModule, **predict_kwargs: Any) -> np_dict_type:
        pass

    @abstractmethod
    def save(self, export_folder: str) -> "IPipeline":
        pass

    @staticmethod
    @abstractmethod
    def load(export_folder: str) -> "IPipeline":
        pass


class ModelSoupConfigs(NamedTuple):
    loader: IDataLoader
    metric_names: Union[str, List[str]]
    metric_configs: configs_type = None
    metric_weights: Optional[Dict[str, float]] = None
    valid_portion: float = 1.0
    strategy: str = "greedy"
    states_callback: states_callback_type = None
    verbose: bool = True


def _generate_model_soup_checkpoint(
    m: "DLPipeline",
    checkpoint_folder: str,
    configs: ModelSoupConfigs,
    scores_export_path: str,
    checkpoint_export_path: str,
) -> None:
    current_states: tensor_dict_type = {}
    with open(os.path.join(checkpoint_folder, SCORES_FILE), "r") as f:
        scores = json.load(f)
    sorted_files = get_sorted_checkpoints(checkpoint_folder)
    sorted_paths = [os.path.join(checkpoint_folder, file) for file in sorted_files]
    metrics = _IMetric.fuse(
        configs.metric_names,
        configs.metric_configs,
        metric_weights=configs.metric_weights,
    )
    chosen_ingredients = []
    if configs.strategy == "greedy":
        n = 0
        best_score = -math.inf
        if configs.verbose:
            print("\n".join(["=" * 100, "Creating Model Soup", "-" * 100]))
        for file, path in zip(sorted_files, sorted_paths):
            if configs.verbose:
                print(f"> Checking {file}")
            states = torch.load(path, map_location=m.device)
            m._make_modifier().permute_states(states)
            states_backup = shallow_copy_dict(current_states)
            if configs.states_callback is not None:
                states = configs.states_callback(m, states)
            for k, v in states.items():
                if n == 0:
                    current_states[k] = v
                else:
                    current_v = current_states[k]
                    current_states[k] = (current_v * n + v) / (n + 1)
            m.model.load_state_dict(current_states)
            kw = dict(
                portion=configs.valid_portion,
                metrics=metrics,
            )
            if check_requires(m.inference.get_outputs, "use_loader_cache"):
                kw["use_loader_cache"] = False
            res = m.inference.get_outputs(configs.loader, **kw)  # type: ignore
            score = res.metric_outputs.final_score  # type: ignore
            if score < best_score:
                current_states = states_backup
                if configs.verbose:
                    print(">> Performance is not improving")
            else:
                n += 1
                best_score = score
                if configs.verbose:
                    chosen_ingredients.append(f"{file} ({scores[file]})")
                    print(f">> New SOTA! ({score})")
    else:
        msg = f"model soup strategy `{configs.strategy}` is not implemented yet"
        raise NotImplementedError(msg)
    with open(scores_export_path, "r") as rf:
        existing_scores = json.load(rf)
    key = list(existing_scores.keys())[0]
    with open(scores_export_path, "w") as wf:
        json.dump({key: best_score}, wf)
    torch.save(current_states, checkpoint_export_path)
    original_metric_names = m.trainer_config["metric_names"]
    if original_metric_names is None:
        metrics_identifier = "+ DEFAULTS +"
    else:
        if isinstance(original_metric_names, str):
            original_metric_names = [original_metric_names]
        metrics_identifier = " | ".join(original_metric_names)
    if configs.verbose:
        print(
            "\n".join(
                [
                    "=" * 100,
                    "Model Soup Generated",
                    "-" * 100,
                    f"Ingredients (Original Metrics : {metrics_identifier})",
                    "-" * 100,
                    "\n".join(chosen_ingredients),
                    "-" * 100,
                    "Final Score",
                    "-" * 100,
                    str(best_score),
                    "-" * 100,
                ]
            )
        )


class IDLPipeline:
    modifier: str

    _defaults: OrderedDictType[str, Any]

    data: Optional[DLDataModule]
    data_info: Dict[str, Any]
    data_module_bytes: Optional[bytes]
    loss: ILoss
    loss_name: str
    loss_config: Optional[Dict[str, Any]]
    model: IDLModel
    model_name: str
    model_config: Dict[str, Any]
    trainer: Trainer
    trainer_config: Dict[str, Any]
    inference: IInference
    inference_base: Type[IInference]
    device_info: DeviceInfo

    pipeline_key: str
    pipeline_file: str
    config_name: str
    trainer_config_file: str
    data_info_name: str
    metrics_log_file: str

    final_results_file: str
    config_bundle_name: str

    config: Dict[str, Any]
    input_dim: Optional[int]

    built: bool
    in_loading: bool

    device: torch.device
    is_rank_0: bool


def get_requirements(c: Type, *, excludes: List[str]) -> List[str]:
    lines = inspect.getsource(c).split("\n")[1:]
    requirements = list(filter(bool, (line.strip().split(":")[0] for line in lines)))
    for e in excludes:
        requirements.remove(e)
    return requirements


class IModifier(WithRegister["IModifier"], IDLPipeline):
    d = dl_pipeline_modifiers

    build_steps = [
        "record_num_samples",
        "prepare_workplace",
        "prepare_loss",
        "build_model",
        "build_inference",
    ]
    requirements = get_requirements(IDLPipeline, excludes=["data"])

    def __init__(self, pipeline: "DLPipeline") -> None:
        self.__pipeline = pipeline

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.__pipeline, __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        pipeline_key = "_IModifier__pipeline"
        if __name == pipeline_key:
            self.__dict__[pipeline_key] = __value
        else:
            setattr(self.__pipeline, __name, __value)

    def _write_pipeline_info(self, folder: str) -> None:
        with open(os.path.join(folder, self.pipeline_file), "w") as f:
            f.write(self.__pipeline.__identifier__)

    def _sanity_check(self) -> None:
        missing_requirements = []
        token = f"{random_hash()}{random_hash()}{random_hash()}"
        for requirement in self.requirements:
            if getattr(self.__pipeline, requirement, token) is token:
                missing_requirements.append(requirement)
        if missing_requirements:
            pipeline_name = self.__pipeline.__class__.__name__
            raise ValueError(
                f"the following attributes is missing in `{pipeline_name}`: "
                f"{', '.join(missing_requirements)}"
            )

    @staticmethod
    def _report_messages(title: str, messages: Dict[str, Any]) -> None:
        def _stringify_item(item: Tuple[str, Any], prefix: Optional[str] = None) -> str:
            key, value = item
            if prefix is not None:
                key = f"{prefix}{key}"
            if not isinstance(value, dict) or not value:
                return f"{key:>{span}s}   |   {value}"
            prefix = f"{key}."
            items = [_stringify_item((vk, vv), prefix) for vk, vv in value.items()]
            return "\n".join(items)

        span = 64
        length = 2 * span
        print(
            "\n".join(
                [
                    "=" * length,
                    f"{title:^{length}s}",
                    "-" * length,
                    "\n".join(map(_stringify_item, messages.items())),
                    "-" * length,
                ]
            )
        )

    def _report_defaults(self) -> None:
        self._report_messages(
            "Internal Default Configurations Used by `carefree-learn`",
            self._defaults,
        )

    def _report_configs(self) -> None:
        self._report_messages("External Configurations", self.config)

    # build steps

    def record_num_samples(self) -> None:
        data: Optional[DLDataModule] = getattr(self, "data", None)
        if data is None:
            self._defaults["train_samples"] = None
            self._defaults["valid_samples"] = None
        else:
            self._defaults["train_samples"] = len(data.train_data)
            if data.valid_data is None:
                self._defaults["valid_samples"] = None
            else:
                self._defaults["valid_samples"] = len(data.valid_data)
        self._defaults.move_to_end("valid_samples", last=False)
        self._defaults.move_to_end("train_samples", last=False)

    def prepare_workplace(self) -> None:
        if self.is_rank_0 and not self.in_loading:
            workplace = prepare_workplace_from(self.trainer_config["workplace"])
            self._defaults["workplace"] = workplace
            self.trainer_config["workplace"] = workplace
            self.trainer_config["data_info_name"] = self.data_info_name
            self.trainer_config["metrics_log_file"] = self.metrics_log_file
            self._write_pipeline_info(workplace)
            Saving.save_dict(self.config, self.config_name, workplace)

    def prepare_loss(self) -> None:
        if self.in_loading:
            return None
        self.loss_name = ILoss.parse(self.loss_name)
        self.loss = ILoss.make(self.loss_name, self.loss_config or {})

    def build_model(self, data_info: Dict[str, Any]) -> None:
        self.model = IDLModel.make(self.model_name, config=self.model_config)

    def build_inference(self) -> None:
        self.inference = self.inference_base(model=self.model)

    def prepare_trainer_defaults(self, data_info: Dict[str, Any]) -> None:
        # set some trainer defaults to deep learning tasks which work well in practice
        if get_ddp_info() is not None:
            mns = self.trainer_config["monitor_names"]
            if mns is not None and mns != "conservative" and mns != ["conservative"]:
                print_warning(
                    "only `conservative` monitor is available "
                    f"in DDP mode, {mns} found"
                )
            self.trainer_config["monitor_names"] = "conservative"
        if self.trainer_config["monitor_names"] is None:
            self._defaults["monitor_names"] = "conservative"
            self.trainer_config["monitor_names"] = "conservative"
        tqdm_settings = self.trainer_config["tqdm_settings"]
        callback_names = self.trainer_config["callback_names"]
        callback_configs = self.trainer_config["callback_configs"]
        optimizer_settings = self.trainer_config["optimizer_settings"]
        if callback_names is None:
            callback_names = []
        if callback_configs is None:
            callback_configs = {}
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        auto_callback = self.trainer_config.get("auto_callback", True)
        if "mlflow" in callback_names and auto_callback:
            mlflow_config = callback_configs.setdefault("mlflow", {})
            if "experiment_name" not in mlflow_config:
                mlflow_config["experiment_name"] = self.model.__identifier__
                self._defaults["mlflow_experiment_name"] = self.model.__identifier__
        if "_log_metrics_msg" not in callback_names and auto_callback:
            self._defaults["additional_callbacks"] = ["_log_metrics_msg"]
            callback_names.insert(0, "_log_metrics_msg")
            verbose = False
            if tqdm_settings is None or not tqdm_settings.get("use_tqdm", False):
                verbose = True
            log_metrics_msg_config = callback_configs.setdefault("_log_metrics_msg", {})
            if "verbose" not in log_metrics_msg_config:
                log_metrics_msg_config["verbose"] = verbose
                self._defaults["log_metrics_msg_verbose"] = verbose
        self.trainer_config["tqdm_settings"] = tqdm_settings
        self.trainer_config["callback_names"] = callback_names
        self.trainer_config["callback_configs"] = callback_configs
        self.trainer_config["optimizer_settings"] = optimizer_settings

    ## api

    def build(self, data_info: Dict[str, Any]) -> None:
        if self.built:
            return None
        self.data_info = shallow_copy_dict(data_info)
        kw = dict(data_info=data_info)
        for step in self.build_steps:
            safe_execute(getattr(self, step), shallow_copy_dict(kw))
        if self.in_loading:
            return None
        safe_execute(self.prepare_trainer_defaults, shallow_copy_dict(kw))
        trainer_config = shallow_copy_dict(self.trainer_config)
        if isinstance(self.model, ModelWithCustomSteps):
            self.model.permute_trainer_config(trainer_config)
        self.trainer = make_trainer(**trainer_config)
        self._sanity_check()
        if self.trainer.is_rank_0 and not self.trainer.tqdm_settings.in_distributed:
            self._report_defaults()
            self._report_configs()
        self.built = True

    # load steps

    def load_states_from(self, folder: str) -> Dict[str, Any]:
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            raise ValueError(f"no model file found in {folder}")
        checkpoint_path = os.path.join(folder, checkpoints[0])
        return torch.load(checkpoint_path, map_location=self.device)

    # changes should happen inplace
    def permute_states(self, states: Dict[str, Any]) -> None:
        pass

    ## api

    def save_misc(self, export_folder: str) -> float:
        os.makedirs(export_folder, exist_ok=True)
        self._write_pipeline_info(export_folder)
        data: Optional[DLDataModule] = getattr(self, "data", None)
        if data is not None:
            data.save_info(export_folder)
        else:
            if self.data_module_bytes is None:
                print_warning(
                    "`data_module_bytes` is not found. This is likely to cause "
                    "data-processing issues unless you are "
                    "saving a pretrained model."
                )
            else:
                msg = "`data` is not found, `pipeline.data_module_bytes` will be saved"
                print_warning(msg)
                data_folder = os.path.join(export_folder, DLDataModule.package_folder)
                zip_path = f"{data_folder}.zip"
                with open(zip_path, "wb") as f:
                    f.write(self.data_module_bytes)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_folder)
                os.remove(zip_path)
        # final results
        final_results = None
        try:
            final_results = self.trainer.final_results
            if final_results is None:
                print_warning(
                    "`final_results` are not generated yet, "
                    "'unknown' results will be saved"
                )
        except AttributeError as e:
            print_warning(
                f"{e}, so `final_results` cannot be accessed, "
                "and 'unknown' results will be saved"
            )
        if final_results is None:
            final_results = MetricsOutputs(0.0, {"unknown": 0.0})
        with open(os.path.join(export_folder, self.final_results_file), "w") as f:
            json.dump(final_results, f)
        # config bundle
        config_bundle = {
            "config": shallow_copy_dict(self.config),
            "device_info": self.device_info,
        }
        Saving.save_dict(config_bundle, self.config_bundle_name, export_folder)
        return final_results.final_score

    @staticmethod
    def load_infrastructure(
        cls: Type["DLPipeline"],
        export_folder: str,
        cuda: Optional[str],
        to_original_device: bool,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "DLPipeline":
        config_bundle = Saving.load_dict(cls.config_bundle_name, export_folder)
        if pre_callback is not None:
            pre_callback(config_bundle)
        config = config_bundle["config"]
        config["in_loading"] = True
        m = safe_execute(cls, config)
        device_info = DeviceInfo(*config_bundle["device_info"])
        if not to_original_device:
            device_info = device_info._replace(cuda=cuda)
        elif cuda is not None:
            print_warning(
                "`to_original_device` is set to True, so "
                f"`cuda={cuda}` will be ignored"
            )
        m.device_info = device_info
        if post_callback is not None:
            post_callback(m, config_bundle)
        return m

    # inference

    def postprocess(self, outputs: InferenceOutputs) -> np_dict_type:
        return outputs.forward_results


@IPipeline.register("dl")
class DLPipeline(IPipeline, IDLPipeline, metaclass=ConfigMeta):
    modifier = "dl"

    inference_base = IInference

    pipeline_key = "pipeline"
    pipeline_file = "pipeline.txt"
    config_name = "config"
    trainer_config_file = "trainer_config.json"
    data_info_name = "data_info"
    metrics_log_file = "metrics.txt"

    final_results_file = "final_results.json"
    config_bundle_name = "config_bundle"

    def __init__(
        self,
        model_name: str,
        model_config: Optional[Dict[str, Any]] = None,
        *,
        loss_name: Optional[str] = None,
        loss_config: Optional[Dict[str, Any]] = None,
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
        allow_no_loss: bool = False,
    ):
        # record defaults
        self._defaults = OrderedDict()
        # sanity check
        if loss_name is None:
            if model_name in loss_dict:
                loss_name = model_name
            else:
                model_base = IDLModel.get(model_name)
                if allow_no_loss or issubclass(model_base, ModelWithCustomSteps):
                    loss_name = ILoss.placeholder_key
                else:
                    raise ValueError(
                        "`loss_name` should be provided when "
                        f"`{model_name}` has not implemented its own loss "
                        "and `allow_no_loss` is False"
                    )
            self._defaults["loss_name"] = loss_name
        # set defaults
        if state_config is None:
            state_config = {}
        if "max_snapshot_file" not in state_config:
            state_config["max_snapshot_file"] = 25
            self._defaults["max_snapshot_file"] = 25
        if callback_names is None:
            if model_name in callback_dict:
                callback_names = model_name
                self._defaults["callback_names"] = model_name
        # initialize
        self.input_dim = None
        self.model_name = model_name
        self.model_config = model_config or {}
        self.data_module_bytes = None
        self.loss_name = loss_name
        self.loss_config = loss_config
        self.trainer_config: Dict[str, Any] = {
            "state_config": state_config,
            "num_epoch": num_epoch,
            "max_epoch": max_epoch,
            "fixed_epoch": fixed_epoch,
            "fixed_steps": fixed_steps,
            "log_steps": log_steps,
            "valid_portion": valid_portion,
            "amp": amp,
            "clip_norm": clip_norm,
            "metric_names": metric_names,
            "metric_configs": metric_configs,
            "metric_weights": metric_weights,
            "use_losses_as_metrics": use_losses_as_metrics,
            "loss_metrics_weights": loss_metrics_weights,
            "recompute_train_losses_in_eval": recompute_train_losses_in_eval,
            "monitor_names": monitor_names,
            "monitor_configs": monitor_configs,
            "callback_names": callback_names,
            "callback_configs": callback_configs,
            "lr": lr,
            "optimizer_name": optimizer_name,
            "scheduler_name": scheduler_name,
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "update_scheduler_per_epoch": update_scheduler_per_epoch,
            "optimizer_settings": optimizer_settings,
            "use_zero": use_zero,
            "workplace": _get_environ_workplace() or workplace,
            "finetune_config": finetune_config,
            "tqdm_settings": tqdm_settings,
        }
        self.in_loading = in_loading
        self.built = False
        self.device_info = DeviceInfo(None, None)
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, DLPipeline):
            return False
        return json.dumps(self.to_json()) == json.dumps(__o.to_json())

    @property
    def device(self) -> torch.device:  # type: ignore
        return self.device_info.device

    @property
    def is_rank_0(self) -> bool:  # type: ignore
        ddp_info = get_ddp_info()
        if ddp_info is None:
            return True
        if ddp_info.rank == 0:
            return True
        return False

    def _make_modifier(self) -> IModifier:
        return IModifier.make(self.modifier, {"pipeline": self})

    # api

    def build(self, data_info: Dict[str, Any]) -> None:
        self._make_modifier().build(data_info)

    def fit(  # type: ignore
        self,
        data: DLDataModule,
        *,
        sample_weights: sample_weights_type = None,
        cuda: Optional[Union[int, str]] = None,
    ) -> "DLPipeline":
        data.prepare(sample_weights)
        if cuda is not None:
            cuda = str(cuda)
        self.data = data
        self.build(data.info)
        self.trainer.fit(
            data,
            self.loss,
            self.model,
            self.inference,
            config_export_file=self.trainer_config_file,
            cuda=cuda,
        )
        self.device_info = self.trainer.device_info
        return self

    def predict(self, data: DLDataModule, **predict_kwargs: Any) -> np_dict_type:
        train_loader, valid_loader = data.initialize()
        if valid_loader is not None:
            raise ValueError("`valid_loader` should not be provided in `predict`")
        kw0 = shallow_copy_dict(predict_kwargs)
        kw1 = shallow_copy_dict(predict_kwargs)
        kw0["loader"] = train_loader
        outputs = safe_execute(self.inference.get_outputs, kw0)
        kw1["outputs"] = outputs
        return safe_execute(self._make_modifier().postprocess, kw1)

    def save(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        remove_original: bool = True,
    ) -> "DLPipeline":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            score = self._make_modifier().save_misc(export_folder)
            if getattr(self, "trainer", None) is not None:
                if getattr(self.trainer, "model", None) is None:
                    self.trainer.model = self.model
                self.trainer.save_checkpoint(score, export_folder, no_history=True)
            else:
                file = f"{PT_PREFIX}-1.pt"
                torch.save(self.model.state_dict(), os.path.join(export_folder, file))
                with open(os.path.join(export_folder, SCORES_FILE), "w") as f:
                    json.dump({file: 0.0}, f)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def pack(
        cls,
        workplace: str,
        *,
        step: Optional[str] = None,
        config_bundle_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        pack_folder: Optional[str] = None,
        cuda: Optional[Union[int, str]] = None,
        compress: bool = True,
        model_soup_configs: Optional[ModelSoupConfigs] = None,
    ) -> str:
        if cuda is not None:
            cuda = str(cuda)
        if pack_folder is None:
            pack_name = f"packed{'' if step is None else f'_{step}'}"
            pack_folder = os.path.join(workplace, pack_name)
        if os.path.isdir(pack_folder):
            print_warning(f"'{pack_folder}' already exists, it will be erased")
            shutil.rmtree(pack_folder)
        os.makedirs(pack_folder)
        abs_folder = os.path.abspath(pack_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [pack_folder]):
            shutil.copyfile(
                os.path.join(workplace, cls.pipeline_file),
                os.path.join(pack_folder, cls.pipeline_file),
            )
            checkpoint_folder = os.path.join(workplace, CHECKPOINTS_FOLDER)
            if step is not None:
                best_file = f"{PT_PREFIX}{step}.pt"
            else:
                best_file = get_sorted_checkpoints(checkpoint_folder)[0]
            new_file = f"{PT_PREFIX}-1.pt"
            new_path = os.path.join(pack_folder, new_file)
            shutil.copyfile(os.path.join(checkpoint_folder, best_file), new_path)
            with open(os.path.join(checkpoint_folder, SCORES_FILE), "r") as rf:
                scores = json.load(rf)
            new_scores_path = os.path.join(pack_folder, SCORES_FILE)
            with open(new_scores_path, "w") as wf:
                json.dump({new_file: scores[best_file]}, wf)
            config = Saving.load_dict(cls.config_name, workplace)
            config_bundle = {
                "config": config,
                "device_info": DeviceInfo(cuda, None),
            }
            if config_bundle_callback is not None:
                config_bundle_callback(config_bundle)
            Saving.save_dict(config_bundle, cls.config_bundle_name, pack_folder)
            shutil.copytree(
                os.path.join(workplace, DataModule.package_folder),
                os.path.join(pack_folder, DataModule.package_folder),
            )
        if model_soup_configs is not None:
            m = DLPipeline.load(pack_folder, cuda=cuda, compress=False)
            _generate_model_soup_checkpoint(
                m,
                checkpoint_folder,
                model_soup_configs,
                new_scores_path,
                new_path,
            )
        if compress:
            with lock_manager(base_folder, [pack_folder]):
                Saving.compress(abs_folder, remove_original=True)
        return pack_folder

    @staticmethod
    def get_base(workplace: str) -> Type["DLPipeline"]:
        with open(os.path.join(workplace, DLPipeline.pipeline_file), "r") as f:
            return DLPipeline.get(f.read())  # type: ignore

    @staticmethod
    def load(
        export_folder: str,
        *,
        cuda: Optional[Union[int, str]] = None,
        to_original_device: bool = False,
        compress: bool = True,
        states_callback: states_callback_type = None,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "DLPipeline":
        if export_folder.endswith(".zip"):
            export_folder = export_folder[:-4]
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                pipeline_cls = DLPipeline.get_base(export_folder)
                m = IModifier.load_infrastructure(
                    pipeline_cls,
                    export_folder,
                    None if cuda is None else str(cuda),
                    to_original_device,
                    pre_callback,
                    post_callback,
                )
                try:
                    data_base = DataModule._get_base(export_folder).base
                    m.data_type = data_base.__identifier__
                    loaded = DataModule.load_info(export_folder, return_bytes=True)
                    data_info = loaded.info
                    m.data_module_bytes = loaded.data_module_bytes
                except Exception as err:
                    print_warning(
                        "error occurred when trying to load "
                        f"`DataModule` ({err}), it might cause by BC breaking, "
                        "empty `data_info` will be used"
                    )
                    data_info = {}
                modifier = m._make_modifier()
                modifier.build(data_info)
                m.model.to(m.device)
                # restore checkpoint
                states = modifier.load_states_from(export_folder)
                modifier.permute_states(states)
                if states_callback is not None:
                    states = states_callback(m, states)
                m.model.load_state_dict(states)
        return m

    def to_onnx(
        self,
        export_folder: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        onnx_file: str = "model.onnx",
        opset: int = 11,
        simplify: bool = True,
        forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_names: Optional[List[str]] = None,
        input_sample: Optional[tensor_dict_type] = None,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "DLPipeline":
        if input_sample is None:
            if getattr(self, "trainer", None) is None:
                msg = "either `input_sample` or `trainer` should be provided"
                raise ValueError(msg)
            input_sample = self.trainer.input_sample
            input_sample.pop(BATCH_INDICES_KEY, None)
        assert isinstance(input_sample, dict)
        self.model.to_onnx(
            export_folder,
            input_sample,
            dynamic_axes,
            onnx_file=onnx_file,
            opset=opset,
            simplify=simplify,
            forward_fn=forward_fn,
            output_names=output_names,
            num_samples=num_samples,
            verbose=verbose,
            **kwargs,
        )
        return self

    @classmethod
    def pack_onnx(
        cls,
        workplace: str,
        export_folder: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        input_sample: tensor_dict_type,
        step: Optional[str] = None,
        config_bundle_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        pack_folder: Optional[str] = None,
        states_callback: states_callback_type = None,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
        onnx_file: str = "model.onnx",
        opset: int = 11,
        simplify: bool = True,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "DLPipeline":
        packed = cls.pack(
            workplace,
            step=step,
            config_bundle_callback=config_bundle_callback,
            pack_folder=pack_folder,
        )
        m = cls.load(
            packed,
            states_callback=states_callback,
            pre_callback=pre_callback,
            post_callback=post_callback,
        )
        m.to_onnx(
            export_folder,
            dynamic_axes,
            onnx_file=onnx_file,
            opset=opset,
            simplify=simplify,
            input_sample=input_sample,
            num_samples=num_samples,
            verbose=verbose,
            **kwargs,
        )
        return m

    def to_json(self, *, export_path: Optional[str] = None) -> Dict[str, Any]:
        d = shallow_copy_dict(self.config)
        d[self.pipeline_key] = self.__identifier__
        if export_path is not None:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(d, f)
        return d

    @classmethod
    def from_json(cls, d: Union[str, Dict[str, Any]]) -> "DLPipeline":
        if isinstance(d, dict):
            d = shallow_copy_dict(d)
        elif isinstance(d, str):
            with open(d, "r") as f:
                d = json.load(f)
        else:
            raise ValueError(f"unrecognized input {d} occurred")
        assert isinstance(d, dict)
        name = d.pop(cls.pipeline_key)
        return cls.make(name, d)


def register_modifier(name: str, *, allow_duplicate: bool = False) -> Callable:
    return IModifier.register(name, allow_duplicate=allow_duplicate)


register_modifier("dl")(IModifier)


__all__ = [
    "register_modifier",
    "IModifier",
    "IPipeline",
    "DLPipeline",
]
