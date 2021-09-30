from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

from ...data import DLLoader
from ...data import DLDataModule
from ...trainer import callback_dict
from ...pipeline import DLPipeline
from ...protocol import loss_dict
from ...protocol import ModelProtocol
from ...protocol import InferenceProtocol
from ...misc.toolkit import get_arguments


@DLPipeline.register("dl.simple")
class SimplePipeline(DLPipeline):
    inference: InferenceProtocol
    inference_base = InferenceProtocol

    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        *,
        loss_name: str,
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
        metric_configs: Optional[Dict[str, Any]] = None,
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
        workplace: str = "_logs",
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
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
            workplace=workplace,
            finetune_config=finetune_config,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
        )
        self.input_dim = None
        self.model_name = model_name
        self.model_config = model_config or {}

    def _prepare_modules(self, data_info: Dict[str, Any]) -> None:
        self._prepare_workplace()
        self._prepare_loss()
        self.model = ModelProtocol.make(self.model_name, config=self.model_config)
        self.inference = self.inference_base(model=self.model)

    def _make_new_loader(  # type: ignore
        self,
        data: DLDataModule,
        batch_size: int = 0,
        **kwargs: Any,
    ) -> DLLoader:
        train_loader, valid_loader = data.initialize()
        if valid_loader is not None:
            raise ValueError("`valid_loader` should not be provided")
        assert isinstance(train_loader, DLLoader)
        return train_loader


@DLPipeline.register("dl.carefree")
class CarefreePipeline(SimplePipeline):
    def __init__(
        self,
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
        metric_configs: Optional[Dict[str, Any]] = None,
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
        workplace: str = "_logs",
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
    ):
        if loss_name is None:
            loss_name = model_name if model_name in loss_dict else "mse"
        if state_config is None:
            state_config = {}
        state_config.setdefault("max_snapshot_file", 25)
        if callback_names is None:
            if model_name in callback_dict:
                callback_names = model_name
        super().__init__(
            model_name,
            model_config,
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
            workplace=workplace,
            finetune_config=finetune_config,
            tqdm_settings=tqdm_settings,
            in_loading=in_loading,
        )

    def _prepare_trainer_defaults(self, data_info: Dict[str, Any]) -> None:
        if self.trainer_config["monitor_names"] is None:
            self.trainer_config["monitor_names"] = "conservative"
        super()._prepare_trainer_defaults(data_info)


__all__ = [
    "SimplePipeline",
    "CarefreePipeline",
]
