import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from torch.optim import Optimizer
from cftool.types import tensor_dict_type
from torch.cuda.amp import autocast

from ...schema import IDLModel
from ...schema import ITrainer
from ...schema import IDataLoader
from ...schema import StepOutputs
from ...schema import TrainerState
from ...schema import PrecisionType
from ...schema import TrainerConfig
from ...schema import MetricsOutputs
from ...constants import LOSS_KEY
from ...data.utils import TensorBatcher
from ...misc.toolkit import get_device
from ...misc.toolkit import no_grad_context
from ...misc.toolkit import toggle_optimizer


def get_update_fn(trainer: ITrainer) -> Callable[[Tensor, Optimizer, bool], None]:
    def update_fn(loss: Tensor, optimizer: Optimizer, update: bool) -> None:
        accelerator = trainer.accelerator
        accelerator.backward(loss)
        if update:
            trainer.clip_norm_step()
            optimizer.step()
            optimizer.zero_grad()

    return update_fn


def run_train_steps(
    m: "ModelWithCustomSteps",
    train_steps: List["CustomTrainStep"],
    *,
    batch_idx: int,
    batch: tensor_dict_type,
    trainer: "ITrainer",
    forward_kwargs: Dict[str, Any],
    loss_kwargs: Dict[str, Any],
) -> StepOutputs:
    """
    Runs a series of custom training steps on a given model and batch of data.

    Parameters
    ----------
    m : ModelWithCustomSteps
        The model to train.
    train_steps : List[CustomTrainStep]
        The custom training steps to run.
    batch_idx : int
        The current batch index.
    batch : tensor_dict_type
        The batch of data to use for training.
    trainer : ITrainer
        The trainer object used to train the model.
    forward_kwargs : Dict[str, Any]
        Additional arguments to pass to the forward pass of the model.
    loss_kwargs : Dict[str, Any]
        Additional arguments to pass to the loss function of each training step.

    Returns
    -------
    StepOutputs
        An object containing the outputs of the forward pass and the calculated loss values of the training steps.

    Step by step explanation
    ------------------------
    1. Initialize variables: `forward` (an empty dictionary), `loss_dict` (an empty dictionary), `any_update`
    (a bool flag set to `False`), `performed_scheduler_step` (a bool flag set to `False`), and `update_fn` (a
    function returned by the `get_update_fn` function defined above).
    2. Check whether the forward pass should have gradients (`fw_has_grad`) and which training step to use for the
    forward pass (`fw_train_step`). This is done by looping through each training step and checking its
    `requires_new_forward` and `requires_grad_in_forward` attributes.
    3. If `fw_has_grad` is `False` and a subsequent training step requires gradients in the forward pass, raise a
    ValueError with a message indicating which training steps have conflicting requirements.
    4. Loop through each training step and execute the following steps for each:
      1) Check whether the current training step should be skipped. If so, move on to the next training step.
      2) If this is the first training step, or if `requires_new_forward` is `True` for the current training step,
      execute the forward pass of the model and store the output in `forward`. The `no_grad_context` context manager
      is used to prevent gradients from being calculated if `requires_grad_in_forward` is `False`.
      3) Get the optimizer to be used for this training step.
      4) If `enable_toggle_optimizer` is `True` for this training step, temporarily switch to the optimizer associated
      with this training step using the `toggle_optimizer` context manager.
      5) Calculate the loss for this training step using the model, state, batch, and forward pass outputs. The
      `autocast` context manager is used if mixed-precision training is enabled.
      6) Update the optimizer if `train_step.grad_accumulate` is a factor of the current `state.step`.
      7) Update the `loss_dict` with the loss values for this training step.
      8) If an optimizer update occurred, set `any_update` to `True`, and if `requires_scheduler_step` is `True` for
      this training step, call `trainer.scheduler_step()` to update the learning rate.
    5. If any optimizer updates occurred but no scheduler steps were performed, call `trainer.scheduler_step()` to
    update the learning rate.
    6. Loop through each training step and call its callback function with the model, trainer, batch, and forward pass outputs.
    7. Return the `StepOutputs` object containing the forward pass outputs and loss values.
    """

    state = trainer.state
    forward: Union[tensor_dict_type, List[tensor_dict_type]] = {}
    loss_dict = {}
    update_fn = get_update_fn(trainer)
    any_update = False
    performed_scheduler_step = False
    # sanity check
    fw_has_grad = True
    fw_train_step: Any = ()
    for i, train_step in enumerate(train_steps):
        if i == 0 or train_step.requires_new_forward:
            fw_has_grad = train_step.requires_grad_in_forward
            fw_train_step = train_step
        if not fw_has_grad and train_step.requires_grad_in_forward:
            fw_name = fw_train_step.__class__.__name__
            current_name = train_step.__class__.__name__
            raise ValueError(
                f"current forward pass comes from '{fw_name}' and has no grad, "
                f"but '{current_name}' requires grad in forward. You can either set "
                f"`requires_grad_in_forward` of '{fw_name}' to True, or set "
                f"`requires_new_forward` of '{current_name}' to True."
            )
    # run train steps
    get_fw = lambda: m.run(batch_idx, batch, state, **forward_kwargs)
    for i, train_step in enumerate(train_steps):
        if train_step.should_skip(m, state):
            continue
        if i == 0 or train_step.requires_new_forward:
            with no_grad_context(enabled=not train_step.requires_grad_in_forward):
                if train_step.num_forward == 1:
                    forward = get_fw()
                else:
                    forward = [get_fw() for _ in range(train_step.num_forward)]
        optimizer = trainer.optimizers[train_step.scope]
        with toggle_optimizer(m, optimizer, enabled=train_step.enable_toggle_optimizer):
            with autocast(enabled=trainer.config.mixed_precision != PrecisionType.NO):
                loss_res = train_step.loss_fn(m, state, batch, forward, **loss_kwargs)
            update = state.step % train_step.grad_accumulate == 0
            update_fn(loss_res.loss, optimizer, update)
            loss_dict.update(loss_res.losses)
        if update:
            any_update = True
            performed_scheduler_step = train_step.requires_scheduler_step
            if performed_scheduler_step:
                trainer.scheduler_step()
    if any_update and not performed_scheduler_step:
        trainer.scheduler_step()
    # callbacks
    for train_step in train_steps:
        train_step.callback(m, trainer, batch, forward)
    return StepOutputs(forward, loss_dict)


def weighted_loss_score(config: TrainerConfig, loss_items: Dict[str, float]) -> float:
    if not config.loss_metrics_weights:
        loss = loss_items.get(LOSS_KEY)
        if loss is not None:
            return -loss
        return -sum(loss_items.values()) / len(loss_items)
    score = 0.0
    for k, w in config.loss_metrics_weights.items():
        v = loss_items.get(k)
        if v is None:
            continue
        score -= v * w
    return score


class CustomTrainStepLoss(NamedTuple):
    loss: Tensor
    losses: Dict[str, float]


class CustomTrainStep(ABC):
    def __init__(
        self,
        scope: str = "all",
        *,
        num_forward: int = 1,
        grad_accumulate: int = 1,
        requires_new_forward: bool = False,
        requires_grad_in_forward: bool = True,
        requires_scheduler_step: bool = False,
        enable_toggle_optimizer: bool = True,
    ) -> None:
        self.scope = scope
        self.num_forward = num_forward
        self.grad_accumulate = grad_accumulate
        self.requires_new_forward = requires_new_forward
        self.requires_grad_in_forward = requires_grad_in_forward
        self.requires_scheduler_step = requires_scheduler_step
        self.enable_toggle_optimizer = enable_toggle_optimizer

    # abstract

    @abstractmethod
    def loss_fn(
        self,
        m: "ModelWithCustomSteps",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        pass

    # optional callbacks

    def should_skip(self, m: "ModelWithCustomSteps", state: TrainerState) -> bool:
        return False

    def callback(
        self,
        m: "ModelWithCustomSteps",
        trainer: "ITrainer",
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
    ) -> None:
        pass


class ModelWithCustomSteps(IDLModel, metaclass=ABCMeta):
    custom_train_step: bool = True
    custom_evaluate_step: bool = True
    custom_params_groups: bool = False
    custom_ddp_initialization: bool = False

    # abstract

    @property
    @abstractmethod
    def train_steps(self) -> List[CustomTrainStep]:
        pass

    @abstractmethod
    def evaluate(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"],
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        forward_kwargs: Dict[str, Any],
    ) -> MetricsOutputs:
        pass

    # inheritance

    def permute_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        if trainer_config.optimizer_settings is None:
            opt_settings = {}
            for step in self.train_steps:
                scope = step.scope
                opt_settings[scope] = dict(
                    optimizer=trainer_config.optimizer_name or "adam",
                    scheduler=trainer_config.scheduler_name,
                    optimizer_config=trainer_config.optimizer_config,
                    scheduler_config=trainer_config.scheduler_config,
                )
            trainer_config.optimizer_settings = opt_settings

    # optional callback

    def params_groups(self, m: nn.Module) -> List[Dict[str, Any]]:
        return []

    # api

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: "ITrainer",
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        return run_train_steps(
            self,
            self.train_steps,
            batch_idx=batch_idx,
            batch=batch,
            trainer=trainer,
            forward_kwargs=forward_kwargs,
            loss_kwargs=loss_kwargs,
        )

    def evaluate_step(
        self,
        config: "TrainerConfig",
        loader: IDataLoader,
        portion: float,
        state: Optional["TrainerState"],
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> MetricsOutputs:
        loss_score_fn = lambda loss_items: weighted_loss_score(config, loss_items)
        device = get_device(self)
        tensor_batcher = TensorBatcher(loader, device)
        is_positive: Dict[str, bool] = {}
        metric_values: Dict[str, List[float]] = {}
        final_scores = []
        for i, batch in enumerate(tensor_batcher):
            if i / len(tensor_batcher) >= portion:
                break
            out = self.evaluate(i, batch, state, loss_score_fn, forward_kwargs or {})
            final_scores.append(out.final_score)
            for k, v in out.metric_values.items():
                metric_values.setdefault(k, []).append(v)
                is_positive[k] = out.is_positive[k]
        return MetricsOutputs(
            sum(final_scores) / len(final_scores),
            {k: sum(v) / len(v) for k, v in metric_values.items()},
            is_positive,
        )


__all__ = [
    "CustomTrainStepLoss",
    "CustomTrainStep",
    "ModelWithCustomSteps",
]
