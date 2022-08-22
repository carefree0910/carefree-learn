import os
import sys
import json

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import update_dict
from cftool.misc import parse_config
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from ..models import *
from .cv.pipeline import CVPipeline
from .ml.api import repeat_with
from .ml.api import RepeatResult
from .ml.pipeline import MLPipeline
from .ml.pipeline import MLCarefreePipeline
from .zoo.core import _parse_config
from .zoo.core import configs_root
from .zoo.core import DLZoo
from ..data import MLData
from ..data import IMLData
from ..data import CVDataModule
from ..data import MLCarefreeData
from ..types import data_type
from ..types import configs_type
from ..types import general_config_type
from ..types import sample_weights_type
from ..types import states_callback_type
from ..pipeline import DLPipeline
from ..pipeline import ModelSoupConfigs
from ..protocol import loss_dict
from ..protocol import metric_dict
from ..protocol import ILoss
from ..protocol import IDLModel
from ..protocol import _IMetric
from ..protocol import IDataLoader
from ..constants import DEFAULT_ZOO_TAG
from ..misc.toolkit import inject_debug
from ..misc.toolkit import download_model
from ..models.protocols.ml import ml_core_dict


# dl


def make(name: str, *, config: general_config_type = None) -> DLPipeline:
    m = DLPipeline.make(name, parse_config(config))
    assert isinstance(m, DLPipeline)
    return m


def run_ddp(path: str, cuda_list: List[Union[int, str]], **kwargs: Any) -> None:
    def _convert_config() -> str:
        return " ".join([f"--{k}={v}" for k, v in kwargs.items()])

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_list))
    kwargs["nproc_per_node"] = len(cuda_list)
    prefix = f"{sys.executable} -m torch.distributed.run "
    os.system(f"{prefix}{_convert_config()} {path}")


def pack(
    workplace: str,
    *,
    step: Optional[str] = None,
    config_bundle_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    pack_folder: Optional[str] = None,
    cuda: Optional[Union[int, str]] = None,
    compress: bool = True,
    # model soup
    model_soup_loader: Optional[IDataLoader] = None,
    model_soup_metric_names: Optional[Union[str, List[str]]] = None,
    model_soup_metric_configs: configs_type = None,
    model_soup_metric_weights: Optional[Dict[str, float]] = None,
    model_soup_valid_portion: float = 1.0,
    model_soup_strategy: str = "greedy",
    model_soup_states_callback: states_callback_type = None,
    model_soup_verbose: bool = True,
) -> str:
    cls = DLPipeline.get_base(workplace)
    if model_soup_loader is None or model_soup_metric_names is None:
        model_soup_configs = None
    else:
        model_soup_configs = ModelSoupConfigs(
            model_soup_loader,
            model_soup_metric_names,
            model_soup_metric_configs,
            model_soup_metric_weights,
            model_soup_valid_portion,
            model_soup_strategy,
            model_soup_states_callback,
            model_soup_verbose,
        )
    return cls.pack(
        workplace,
        step=step,
        config_bundle_callback=config_bundle_callback,
        pack_folder=pack_folder,
        cuda=cuda,
        compress=compress,
        model_soup_configs=model_soup_configs,
    )


def load(
    export_folder: str,
    *,
    cuda: Optional[Union[int, str]] = None,
    compress: bool = True,
    states_callback: states_callback_type = None,
    pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    post_callback: Optional[Callable[[DLPipeline, Dict[str, Any]], None]] = None,
) -> DLPipeline:
    return DLPipeline.load(
        export_folder,
        cuda=cuda,
        compress=compress,
        states_callback=states_callback,
        pre_callback=pre_callback,
        post_callback=post_callback,
    )


def pack_onnx(
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
    post_callback: Optional[Callable[[DLPipeline, Dict[str, Any]], None]] = None,
    onnx_file: str = "model.onnx",
    opset: int = 11,
    simplify: bool = True,
    num_samples: Optional[int] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> DLPipeline:
    cls = DLPipeline.get_base(workplace)
    return cls.pack_onnx(
        workplace,
        export_folder,
        dynamic_axes,
        step=step,
        config_bundle_callback=config_bundle_callback,
        pack_folder=pack_folder,
        states_callback=states_callback,
        pre_callback=pre_callback,
        post_callback=post_callback,
        onnx_file=onnx_file,
        opset=opset,
        simplify=simplify,
        input_sample=input_sample,
        num_samples=num_samples,
        verbose=verbose,
        **kwargs,
    )


def from_json(d: Union[str, Dict[str, Any]]) -> DLPipeline:
    return DLPipeline.from_json(d)


def make_model(name: str, **kwargs: Any) -> IDLModel:
    return IDLModel.make(name, kwargs)


def make_loss(name: str, **kwargs: Any) -> ILoss:
    return ILoss.make(name, kwargs)


def make_metric(name: str, **kwargs: Any) -> _IMetric:
    return _IMetric.make(name, kwargs)


class ModelItem(NamedTuple):
    name: str
    requirements: Dict[str, Any]


def model_zoo(*, verbose: bool = False) -> List[ModelItem]:
    def _squeeze_requirements(req: Dict[str, Any], d: Dict[str, Any]) -> None:
        for k, v in req.items():
            kd = d.get(k)
            if kd is None:
                continue
            if isinstance(v, dict):
                _squeeze_requirements(v, kd)
                continue
            assert isinstance(v, list)
            pop_indices = []
            for i, vv in enumerate(v):
                if vv in kd:
                    pop_indices.append(i)
            for i in pop_indices[::-1]:
                v.pop(i)

    models = []
    for task in sorted(os.listdir(configs_root)):
        if task == "common_":
            continue
        task_folder = os.path.join(configs_root, task)
        for model in sorted(os.listdir(task_folder)):
            model_folder = os.path.join(task_folder, model)
            for config_file in sorted(os.listdir(model_folder)):
                config_path = os.path.join(model_folder, config_file)
                d = _parse_config(config_path)
                requirements = d.pop("__requires__", {})
                _squeeze_requirements(requirements, d)
                tag = os.path.splitext(config_file)[0]
                name = f"{task}/{model}"
                if tag != DEFAULT_ZOO_TAG:
                    name = f"{name}.{tag}"
                models.append(ModelItem(name, requirements))
    if verbose:

        def _stringify_item(item: ModelItem) -> str:
            return f"{item.name:>{span}s}   |   {json.dumps(item.requirements)}"

        span = 42
        print(
            "\n".join(
                [
                    "=" * 120,
                    f"{'Names':>{span}s}   |   Requirements",
                    "-" * 120,
                    "\n".join(map(_stringify_item, models)),
                    "-" * 120,
                ]
            )
        )
    return models


def from_zoo(
    model: str,
    *,
    return_model: bool = False,
    no_build: bool = False,
    **kwargs: Any,
) -> Union[IDLModel, DLPipeline]:
    if return_model and no_build:
        raise ValueError("`no_build` should be False when `return_model` is True")
    kwargs["no_build"] = no_build
    fn = DLZoo.load_model if return_model else DLZoo.load_pipeline
    return fn(model, **kwargs)  # type: ignore


def supported_losses() -> List[str]:
    return sorted(loss_dict)


def supported_metrics() -> List[str]:
    return sorted(metric_dict)


# ml


def _make_ml_data(
    x_train: Union[data_type, IMLData],
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    train_others: Optional[np_dict_type] = None,
    valid_others: Optional[np_dict_type] = None,
    carefree: bool = False,
    is_classification: Optional[bool] = None,
    data_config: Optional[Dict[str, Any]] = None,
    cf_data_config: Optional[Dict[str, Any]] = None,
) -> IMLData:
    if isinstance(x_train, IMLData):
        x_train.is_classification = is_classification
        return x_train
    data_kwargs: Dict[str, Any] = {
        "is_classification": is_classification,
        "train_others": train_others,
        "valid_others": valid_others,
    }
    if carefree:
        data_kwargs["cf_data_config"] = cf_data_config
    update_dict(data_config or {}, data_kwargs)
    args = x_train, y_train, x_valid, y_valid
    fn = MLCarefreeData.make_with if carefree else MLData
    return fn(*args, **data_kwargs)  # type: ignore


def fit_ml(
    x_train: Union[data_type, IMLData],
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    *,
    train_others: Optional[np_dict_type] = None,
    valid_others: Optional[np_dict_type] = None,
    # data
    carefree: bool = False,
    is_classification: Optional[bool] = None,
    data_config: Optional[Dict[str, Any]] = None,
    cf_data_config: Optional[Dict[str, Any]] = None,
    # pipeline
    core_name: str = "fcnn",
    core_config: Optional[Dict[str, Any]] = None,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    loss_name: str = "auto",
    loss_config: Optional[Dict[str, Any]] = None,
    # encoder
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
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    use_zero: bool = False,
    workplace: str = "_logs",
    finetune_config: Optional[Dict[str, Any]] = None,
    tqdm_settings: Optional[Dict[str, Any]] = None,
    # misc
    pre_process_batch: bool = True,
    debug: bool = False,
    # fit
    sample_weights: sample_weights_type = None,
    cuda: Optional[Union[int, str]] = None,
) -> MLPipeline:
    pipeline_config = dict(
        core_name=core_name,
        core_config=core_config,
        input_dim=input_dim,
        output_dim=output_dim,
        loss_name=loss_name,
        loss_config=loss_config,
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
        optimizer_settings=optimizer_settings,
        use_zero=use_zero,
        workplace=workplace,
        finetune_config=finetune_config,
        tqdm_settings=tqdm_settings,
        pre_process_batch=pre_process_batch,
    )
    if debug:
        inject_debug(pipeline_config)
    fit_kwargs = dict(sample_weights=sample_weights, cuda=cuda)
    m_base = MLCarefreePipeline if carefree else MLPipeline
    data = _make_ml_data(
        x_train,
        y_train,
        x_valid,
        y_valid,
        train_others,
        valid_others,
        carefree,
        is_classification,
        data_config,
        cf_data_config,
    )
    return m_base(**pipeline_config).fit(data, **fit_kwargs)  # type: ignore


def repeat_ml(
    x_train: Union[data_type, IMLData],
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    *,
    train_others: Optional[np_dict_type] = None,
    valid_others: Optional[np_dict_type] = None,
    # repeat
    workplace: str = "_repeat",
    models: Union[str, List[str]] = "fcnn",
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    sequential: Optional[bool] = None,
    num_jobs: int = 1,
    num_repeat: int = 5,
    return_patterns: bool = True,
    compress: bool = True,
    use_tqdm: bool = True,
    available_cuda_list: Optional[List[int]] = None,
    resource_config: Optional[Dict[str, Any]] = None,
    task_meta_kwargs: Optional[Dict[str, Any]] = None,
    to_original_device: bool = False,
    is_fix: bool = False,
    # data
    carefree: bool = False,
    is_classification: Optional[bool] = None,
    data_config: Optional[Dict[str, Any]] = None,
    cf_data_config: Optional[Dict[str, Any]] = None,
    # pipeline
    core_name: str = "fcnn",
    core_config: Optional[Dict[str, Any]] = None,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    loss_name: str = "auto",
    loss_config: Optional[Dict[str, Any]] = None,
    # encoder
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
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    finetune_config: Optional[Dict[str, Any]] = None,
    tqdm_settings: Optional[Dict[str, Any]] = None,
    # misc
    sample_weights: sample_weights_type = None,
    pre_process_batch: bool = True,
    debug: bool = False,
) -> RepeatResult:
    pipeline_config = dict(
        core_name=core_name,
        core_config=core_config,
        input_dim=input_dim,
        output_dim=output_dim,
        loss_name=loss_name,
        loss_config=loss_config,
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
        optimizer_settings=optimizer_settings,
        finetune_config=finetune_config,
        tqdm_settings=tqdm_settings,
        pre_process_batch=pre_process_batch,
    )
    if debug:
        inject_debug(pipeline_config)
    return repeat_with(
        _make_ml_data(
            x_train,
            y_train,
            x_valid,
            y_valid,
            train_others,
            valid_others,
            carefree,
            is_classification,
            data_config,
            cf_data_config,
        ),
        carefree=carefree,
        workplace=workplace,
        models=models,
        model_configs=model_configs,
        predict_config=predict_config,
        sample_weights=sample_weights,
        sequential=sequential,
        num_jobs=num_jobs,
        num_repeat=num_repeat,
        return_patterns=return_patterns,
        compress=compress,
        use_tqdm=use_tqdm,
        available_cuda_list=available_cuda_list,
        resource_config=resource_config,
        task_meta_kwargs=task_meta_kwargs,
        to_original_device=to_original_device,
        is_fix=is_fix,
        **pipeline_config,
    )


def supported_ml_models() -> List[str]:
    return sorted(ml_core_dict)


# cv


def fit_cv(
    data: CVDataModule,
    model_name: str,
    model_config: Dict[str, Any],
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
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    use_zero: bool = False,
    workplace: str = "_logs",
    finetune_config: Optional[Dict[str, Any]] = None,
    tqdm_settings: Optional[Dict[str, Any]] = None,
    # misc
    debug: bool = False,
    # fit
    sample_weights: sample_weights_type = None,
    cuda: Optional[Union[int, str]] = None,
) -> "CVPipeline":
    pipeline_config = dict(
        model_name=model_name,
        model_config=model_config,
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
        optimizer_settings=optimizer_settings,
        use_zero=use_zero,
        workplace=workplace,
        finetune_config=finetune_config,
        tqdm_settings=tqdm_settings,
    )
    if debug:
        inject_debug(pipeline_config)
    fit_kwargs = dict(sample_weights=sample_weights, cuda=cuda)
    return CVPipeline(**pipeline_config).fit(data, **fit_kwargs)  # type: ignore


# clf


def _clf(
    model: str,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]],
    pretrained_name: Optional[str],
    img_size: Optional[int],
    return_model: bool = False,
    **kwargs: Any,
) -> Any:
    if img_size is not None:
        kwargs["img_size"] = img_size
    kwargs["num_classes"] = num_classes
    if pretrained_name is not None:
        model_config = kwargs.setdefault("model_config", {})
        model_config["encoder1d_pretrained_name"] = pretrained_name
    model = f"clf/{model}"
    fn = DLZoo.load_model if return_model else DLZoo.load_pipeline
    if aux_num_classes is None:
        return fn(model, **kwargs)  # type: ignore
    config = DLZoo(model, no_build=True, **kwargs).config
    aux_labels = sorted(aux_num_classes)
    loss_name = config["loss_name"]
    config["loss_name"] = f"{loss_name}:aux:{','.join(aux_labels)}"
    metric_names = config.setdefault("metric_names", [])
    metric_configs = config.setdefault("metric_configs", {})
    if isinstance(metric_configs, dict):
        metric_configs = [metric_configs.get(name, {}) for name in metric_names]
    for label in aux_labels:
        metric_names.append("aux")
        metric_configs.append({"key": label, "base": "acc"})
    config["metric_configs"] = metric_configs
    model_config = config.setdefault("model_config", {})
    model_config["aux_num_classes"] = aux_num_classes
    return fn(model, **config)  # type: ignore


def cct(
    img_size: int,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _clf("cct", num_classes, aux_num_classes, None, img_size, **kwargs)


def cct_model(
    img_size: int,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> VanillaClassifier:
    return _clf("cct", num_classes, aux_num_classes, None, img_size, True, **kwargs)


def cct_lite(
    img_size: int,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _clf("cct.lite", num_classes, aux_num_classes, None, img_size, **kwargs)


def cct_lite_model(
    img_size: int,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> VanillaClassifier:
    return _clf(
        "cct.lite",
        num_classes,
        aux_num_classes,
        None,
        img_size,
        True,
        **kwargs,
    )


def cct_large(
    img_size: int,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _clf("cct.large", num_classes, aux_num_classes, None, img_size, **kwargs)


def cct_large_model(
    img_size: int,
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> VanillaClassifier:
    return _clf(
        "cct.large",
        num_classes,
        aux_num_classes,
        None,
        img_size,
        True,
        **kwargs,
    )


def cct_large_224(
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> DLPipeline:
    pretrained_name = "cct_large_224" if pretrained else None
    return _clf(
        "cct.large_224",
        num_classes,
        aux_num_classes,
        pretrained_name,
        None,
        **kwargs,
    )


def cct_large_224_model(
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> VanillaClassifier:
    pretrained_name = "cct_large_224" if pretrained else None
    return _clf(
        "cct.large_224",
        num_classes,
        aux_num_classes,
        pretrained_name,
        None,
        True,
        **kwargs,
    )


def cct_large_384(
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> DLPipeline:
    pretrained_name = "cct_large_384" if pretrained else None
    return _clf(
        "cct.large_384",
        num_classes,
        aux_num_classes,
        pretrained_name,
        None,
        **kwargs,
    )


def cct_large_384_model(
    num_classes: int,
    aux_num_classes: Optional[Dict[str, int]] = None,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> VanillaClassifier:
    pretrained_name = "cct_large_384" if pretrained else None
    return _clf(
        "cct.large_384",
        num_classes,
        aux_num_classes,
        pretrained_name,
        None,
        True,
        **kwargs,
    )


def resnet18(num_classes: int, pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_pipeline("clf/resnet18", **kwargs)


def resnet18_model(
    num_classes: int,
    *,
    pretrained: bool = False,
    **kwargs: Any,
) -> VanillaClassifier:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_model("clf/resnet18", **kwargs)


def resnet18_gray(num_classes: int, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    return DLZoo.load_pipeline("clf/resnet18.gray", **kwargs)


def resnet50(num_classes: int, pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_pipeline("clf/resnet50", **kwargs)


def resnet50_model(
    num_classes: int,
    *,
    pretrained: bool = False,
    **kwargs: Any,
) -> VanillaClassifier:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_model("clf/resnet50", **kwargs)


def resnet50_gray(num_classes: int, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    return DLZoo.load_pipeline("clf/resnet50.gray", **kwargs)


def resnet101(num_classes: int, pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_pipeline("clf/resnet101", **kwargs)


def resnet101_model(
    num_classes: int,
    *,
    pretrained: bool = False,
    **kwargs: Any,
) -> VanillaClassifier:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_model("clf/resnet101", **kwargs)


# gan


def vanilla_gan(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/vanilla", **kwargs)


def vanilla_gan_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/vanilla.gray", **kwargs)


def siren_gan(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/siren", **kwargs)


def siren_gan_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/siren.gray", **kwargs)


# generator


def pixel_cnn(num_classes: int, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    return DLZoo.load_pipeline("generator/pixel_cnn", **kwargs)


# multimodal


def clip(pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("multimodal/clip", pretrained=pretrained, **kwargs)


def clip_model(pretrained: bool = True, **kwargs: Any) -> CLIP:
    return DLZoo.load_model("multimodal/clip", pretrained=pretrained, **kwargs)


def clip_vqgan_aligner(**kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("multimodal/clip_vqgan_aligner", **kwargs)


# segmentor


def aim() -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/aim")


def aim_model() -> IDLModel:
    return DLZoo.load_model("segmentor/aim")


def u2net(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/u2net", pretrained=pretrained, **kwargs)


def u2net_model(pretrained: bool = False, **kwargs: Any) -> IDLModel:
    return DLZoo.load_model("segmentor/u2net", pretrained=pretrained, **kwargs)


def u2net_lite(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/u2net.lite", pretrained=pretrained, **kwargs)


def u2net_lite_model(pretrained: bool = False, **kwargs: Any) -> IDLModel:
    return DLZoo.load_model("segmentor/u2net.lite", pretrained=pretrained, **kwargs)


def u2net_finetune(ckpt: Optional[str] = None, **kwargs: Any) -> DLPipeline:
    if ckpt is None:
        ckpt = download_model("u2net")
    kwargs["pretrained_ckpt"] = ckpt
    return DLZoo.load_pipeline("segmentor/u2net.finetune", **kwargs)


def u2net_lite_finetune(ckpt: Optional[str] = None, **kwargs: Any) -> DLPipeline:
    if ckpt is None:
        ckpt = download_model("u2net.lite")
    kwargs["pretrained_ckpt"] = ckpt
    return DLZoo.load_pipeline("segmentor/u2net.finetune_lite", **kwargs)


def u2net_refine(lv1_model_ckpt_path: str, **kwargs: Any) -> DLPipeline:
    kwargs["lv1_model_ckpt_path"] = lv1_model_ckpt_path
    return DLZoo.load_pipeline("segmentor/u2net.refine", **kwargs)


def u2net_lite_refine(lv1_model_ckpt_path: str, **kwargs: Any) -> DLPipeline:
    kwargs["lv1_model_ckpt_path"] = lv1_model_ckpt_path
    return DLZoo.load_pipeline("segmentor/u2net.refine_lite", **kwargs)


# ssl


def dino(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("ssl/dino", **kwargs)


def dino_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["in_channels"] = 1
    return dino(img_size, **kwargs)


# style transfer


def adain(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    kwargs["pretrained"] = pretrained
    return DLZoo.load_pipeline("style_transfer/adain", **kwargs)


# vae


def vanilla_vae(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vanilla", **kwargs)


def vanilla_vae_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vanilla.gray", **kwargs)


def vanilla_vae2d(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vanilla.2d", **kwargs)


def vanilla_vae2d_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vanilla.2d_gray", **kwargs)


def style_vae(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/style", **kwargs)


def style_vae_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/style.gray", **kwargs)


def siren_vae(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/siren", **kwargs)


def siren_vae_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/siren.gray", **kwargs)


def _vq_vae(
    model: str,
    img_size: int,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> DLPipeline:
    kwargs["img_size"] = img_size
    if num_classes is not None:
        model_config = kwargs.setdefault("model_config", {})
        num_classes = model_config.setdefault("num_classes", num_classes)
        callback_names = kwargs.get("callback_names")
        if callback_names is None or "vq_vae" in callback_names:
            callback_configs = kwargs.setdefault("callback_configs", {})
            vq_vae_callback_configs = callback_configs.setdefault("vq_vae", {})
            vq_vae_callback_configs.setdefault("num_classes", num_classes)
    return DLZoo.load_pipeline(f"vae/{model}", **kwargs)


def vq_vae(
    img_size: int,
    *,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _vq_vae("vq", img_size, num_classes, **kwargs)


def vq_vae_gray(
    img_size: int,
    *,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _vq_vae("vq.gray", img_size, num_classes, **kwargs)


def vq_vae_lite(
    img_size: int,
    *,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _vq_vae("vq.lite", img_size, num_classes, **kwargs)


def vq_vae_gray_lite(
    img_size: int,
    *,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> DLPipeline:
    return _vq_vae("vq.gray_lite", img_size, num_classes, **kwargs)


# nlp


def hugging_face(model: str) -> DLPipeline:
    return DLZoo.load_pipeline("hugging_face/general", model_config={"model": model})


def hugging_face_model(model: str) -> HuggingFaceModel:
    return DLZoo.load_model("hugging_face/general", model_config={"model": model})


def simbert_model() -> SimBERT:
    return DLZoo.load_model("hugging_face/simbert")


def opus_model(src: str, tgt: str) -> OPUSBase:
    return DLZoo.load_model("hugging_face/opus", model_config={"src": src, "tgt": tgt})


def opus_zh_en_model() -> OPUS_ZH_EN:
    return opus_model("zh", "en")
