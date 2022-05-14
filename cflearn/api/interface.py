import os
import sys

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import update_dict

from ..models import *
from ..types import data_type
from ..types import configs_type
from ..types import tensor_dict_type
from ..types import general_config_type
from ..types import sample_weights_type
from ..types import states_callback_type
from ..pipeline import DLPipeline
from ..protocol import ModelProtocol
from .cv.pipeline import CarefreePipeline as CVCarefree
from .ml.pipeline import SimplePipeline as MLSimple
from .ml.pipeline import CarefreePipeline as MLCarefree
from .ml.interface import repeat_with
from .ml.interface import RepeatResult
from .zoo.core import DLZoo
from ..data.interface import MLData
from ..data.interface import CVDataModule
from ..misc.toolkit import inject_debug
from ..misc.toolkit import parse_config
from ..misc.toolkit import download_model


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
    cuda: Optional[str] = None,
) -> str:
    cls = DLPipeline.get_base(workplace)
    return cls.pack(
        workplace,
        step=step,
        config_bundle_callback=config_bundle_callback,
        pack_folder=pack_folder,
        cuda=cuda,
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
    compress: Optional[bool] = None,
    remove_original: bool = True,
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
        onnx_only=True,
        input_sample=input_sample,
        num_samples=num_samples,
        compress=compress,
        remove_original=remove_original,
        verbose=verbose,
        **kwargs,
    )


# ml


def _make_ml_data(
    x_train: data_type,
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    carefree: bool = False,
    is_classification: Optional[bool] = None,
    data_config: Optional[Dict[str, Any]] = None,
    cf_data_config: Optional[Dict[str, Any]] = None,
) -> MLData:
    data_kwargs: Dict[str, Any] = {"is_classification": is_classification}
    if carefree:
        data_kwargs["cf_data_config"] = cf_data_config
    update_dict(data_config or {}, data_kwargs)
    args = x_train, y_train, x_valid, y_valid
    data_base = MLData.with_cf_data if carefree else MLData
    return data_base(*args, **data_kwargs)  # type: ignore


def fit_ml(
    x_train: data_type,
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    *,
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
) -> MLSimple:
    pipeline_config = dict(
        core_name=core_name,
        core_config=core_config,
        input_dim=input_dim,
        output_dim=output_dim,
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
    m_base = MLCarefree if carefree else MLSimple
    data = _make_ml_data(
        x_train,
        y_train,
        x_valid,
        y_valid,
        carefree,
        is_classification,
        data_config,
        cf_data_config,
    )
    return m_base(**pipeline_config).fit(data, **fit_kwargs)  # type: ignore


def repeat_ml(
    x_train: data_type,
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    *,
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
            carefree,
            is_classification,
            data_config,
            cf_data_config,
        ),
        pipeline_base=MLCarefree,
        workplace=workplace,
        models=models,
        model_configs=model_configs,
        predict_config=predict_config,
        sequential=sequential,
        num_jobs=num_jobs,
        num_repeat=num_repeat,
        return_patterns=return_patterns,
        compress=compress,
        use_tqdm=use_tqdm,
        available_cuda_list=available_cuda_list,
        resource_config=resource_config,
        task_meta_kwargs=task_meta_kwargs,
        is_fix=is_fix,
        **pipeline_config,
    )


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
) -> "CVCarefree":
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
    return CVCarefree(**pipeline_config).fit(data, **fit_kwargs)  # type: ignore


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


def aim_model() -> ModelProtocol:
    return DLZoo.load_model("segmentor/aim")


def u2net(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/u2net", pretrained=pretrained, **kwargs)


def u2net_model(pretrained: bool = False, **kwargs: Any) -> ModelProtocol:
    return DLZoo.load_model("segmentor/u2net", pretrained=pretrained, **kwargs)


def u2net_lite(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/u2net.lite", pretrained=pretrained, **kwargs)


def u2net_lite_model(pretrained: bool = False, **kwargs: Any) -> ModelProtocol:
    return DLZoo.load_model("segmentor/u2net.lite", pretrained=pretrained, **kwargs)


def u2net_finetune(ckpt: Optional[str] = None, **kwargs: Any) -> DLPipeline:
    if ckpt is None:
        kwargs["pretrained_ckpt"] = download_model("u2net")
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


__all__ = [
    "make",
    "run_ddp",
    "pack",
    "load",
    "pack_onnx",
    "fit_ml",
    "repeat_ml",
    "fit_cv",
    "cct",
    "cct_model",
    "cct_lite",
    "cct_lite_model",
    "cct_large",
    "cct_large_model",
    "cct_large_224",
    "cct_large_224_model",
    "cct_large_384",
    "cct_large_384_model",
    "resnet18",
    "resnet18_model",
    "resnet18_gray",
    "resnet101_model",
    "vanilla_gan",
    "vanilla_gan_gray",
    "siren_gan",
    "siren_gan_gray",
    "pixel_cnn",
    "clip_model",
    "clip_vqgan_aligner",
    "aim",
    "aim_model",
    "u2net",
    "u2net_model",
    "u2net_lite",
    "u2net_lite_model",
    "u2net_finetune",
    "u2net_lite_finetune",
    "u2net_refine",
    "u2net_lite_refine",
    "dino",
    "dino_gray",
    "adain",
    "vanilla_vae",
    "vanilla_vae_gray",
    "vanilla_vae2d",
    "vanilla_vae2d_gray",
    "style_vae",
    "style_vae_gray",
    "siren_vae",
    "siren_vae_gray",
    "vq_vae",
    "vq_vae_gray",
    "vq_vae_lite",
    "vq_vae_gray_lite",
    "hugging_face",
    "hugging_face_model",
]
