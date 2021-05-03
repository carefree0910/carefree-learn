from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

from ...protocol import ModelProtocol
from ..internal_.pipeline import DLPipeline
from ...misc.toolkit import get_arguments
from ...misc.internal_ import DLLoader
from ...misc.internal_ import DLInference


@DLPipeline.register("cv.simple")
class SimplePipeline(DLPipeline):
    inference: DLInference
    inference_base = DLInference

    train_loader: DLLoader
    valid_loader: Optional[DLLoader]

    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        *,
        loss_name: str,
        loss_config: Optional[Dict[str, Any]] = None,
        # data loader
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 4,
        valid_batch_size: int = 4,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
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
            loss_name=loss_name,
            loss_config=loss_config,
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
            loss_metrics_weights=loss_metrics_weights,
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
        self.model_name = model_name
        self.model_config = model_config or {}

    def _prepare_data(self, x: DLLoader, *args: Any) -> None:
        self.train_loader = x
        self.valid_loader = args[0]

    def _prepare_modules(self) -> None:
        self._prepare_workplace()
        self._prepare_loss()
        self.model = ModelProtocol.make(self.model_name, **self.model_config)
        self.inference = DLInference(model=self.model)

    def _make_new_loader(self, x: Any, batch_size: int = 0, **kwargs: Any) -> DLLoader:
        if not isinstance(x, DLLoader):
            msg = "`SimplePipeline` only supports inference with a given loader"
            raise ValueError(msg)
        return x


__all__ = [
    "SimplePipeline",
]
