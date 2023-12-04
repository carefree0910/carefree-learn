import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import Protocol
from torch.optim import Optimizer
from cftool.misc import safe_execute
from cftool.types import tensor_dict_type
from torch.cuda.amp import autocast

from ..losses import build_loss
from ..schema import ILoss
from ..schema import IMetric
from ..schema import DLConfig
from ..schema import IDLModel
from ..schema import ITrainer
from ..schema import TrainStep
from ..schema import IInference
from ..schema import IDataLoader
from ..schema import StepOutputs
from ..schema import TrainerState
from ..schema import PrecisionType
from ..schema import TrainerConfig
from ..schema import MetricsOutputs
from ..schema import TrainStepLoss
from ..schema import forward_results_type
from ..modules import build_module
from ..toolkit import get_clones
from ..toolkit import get_device
from ..toolkit import eval_context
from ..toolkit import no_grad_context
from ..toolkit import toggle_optimizer
from ..trainer import weighted_loss_score
from ..constants import LOSS_KEY
from ..constants import PREDICTIONS_KEY


def get_update_fn(trainer: ITrainer) -> Callable[[Tensor, Optimizer, bool], None]:
    def update_fn(loss: Tensor, optimizer: Optimizer, update: bool) -> None:
        accelerator = trainer.accelerator
        accelerator.backward(loss)
        if update:
            trainer.clip_norm_step()
            optimizer.step()
            optimizer.zero_grad()

    return update_fn


class CommonTrainStep(TrainStep):
    def __init__(self, loss: ILoss):
        super().__init__()
        self.loss = loss

    def loss_fn(
        self,
        m: IDLModel,
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
        **kwargs: Any,
    ) -> TrainStepLoss:
        losses = self.loss.run(forward_results, batch, state)
        return TrainStepLoss(
            losses[LOSS_KEY],
            {k: v.item() for k, v in losses.items()},
        )


@IDLModel.register("common")
class CommonDLModel(IDLModel):
    loss: ILoss

    @property
    def device(self) -> torch.device:
        return get_device(self.m)

    @property
    def train_steps(self) -> List[TrainStep]:
        return [CommonTrainStep(self.loss)]

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m, self.loss]

    def from_accelerator(self, m: nn.Module, loss: nn.Module) -> IDLModel:
        cloned: CommonDLModel = CommonDLModel.from_config(self.config.copy())
        cloned.m = m
        cloned.loss = loss
        return cloned

    def build(self, config: DLConfig) -> None:
        if config.loss_name is None:
            raise ValueError("`loss_name` should be specified for `CommonDLModel`")
        self.m = build_module(config.module_name, config=config.module_config)
        self.loss = build_loss(config.loss_name, config=config.loss_config)

    def step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        forward_kwargs: Optional[Dict[str, Any]] = None,
        *,
        use_grad: bool = False,
        get_losses: bool = False,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> StepOutputs:
        with eval_context(nn.ModuleList(self.all_modules), use_grad=use_grad):
            loss_dict = {}
            loss_kwargs = loss_kwargs or {}
            forward_kwargs = forward_kwargs or {}
            get_fw = lambda: self.run(batch_idx, batch, None, **forward_kwargs)
            train_steps = self.train_steps
            if not train_steps:
                return StepOutputs(get_fw(), {})
            for i, train_step in enumerate(self.train_steps):
                if train_step.should_skip(self, None):
                    continue
                if i == 0 or train_step.requires_new_forward:
                    if train_step.num_forward == 1:
                        fw = get_fw()
                    else:
                        fw = [get_fw() for _ in range(train_step.num_forward)]
                if get_losses:
                    loss_res = train_step.loss_fn(self, None, batch, fw, **loss_kwargs)
                    loss_dict.update(loss_res.losses)
            return StepOutputs(fw, loss_dict)

    def train(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: ITrainer,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        """
        Runs a series of custom training steps on a batch of data.

        Parameters
        ----------
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
        for i, train_step in enumerate(self.train_steps):
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
        get_fw = lambda: self.run(batch_idx, batch, state, **forward_kwargs)
        for i, train_step in enumerate(self.train_steps):
            if train_step.should_skip(self, state):
                continue
            if i == 0 or train_step.requires_new_forward:
                with no_grad_context(enabled=not train_step.requires_grad_in_forward):
                    if train_step.num_forward == 1:
                        forward = get_fw()
                    else:
                        forward = [get_fw() for _ in range(train_step.num_forward)]
            optimizer = trainer.optimizers[train_step.scope]
            with toggle_optimizer(
                self.m, optimizer, enabled=train_step.enable_toggle_optimizer
            ):
                with autocast(
                    enabled=trainer.config.mixed_precision != PrecisionType.NO
                ):
                    loss_res = train_step.loss_fn(
                        self, state, batch, forward, **loss_kwargs
                    )
                grad_period = (
                    train_step.grad_accumulate or trainer.config.grad_accumulate
                )
                update = state.step % grad_period == 0
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
        for train_step in self.train_steps:
            train_step.callback(self, trainer, batch, forward)
        return StepOutputs(forward, loss_dict)

    def evaluate(
        self,
        config: TrainerConfig,
        metrics: Optional[IMetric],
        inference: IInference,
        loader: IDataLoader,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> MetricsOutputs:
        outputs = inference.get_outputs(
            loader,
            portion=portion,
            metrics=metrics,
            use_losses_as_metrics=config.use_losses_as_metrics,  # type: ignore
            return_outputs=False,
            **(forward_kwargs or {}),
        )
        metric_values = {}
        is_positive = {}
        final_scores = []
        loss_items = outputs.loss_items
        metric_outputs = outputs.metric_outputs
        if loss_items is not None:
            metric_values.update(loss_items)
            is_positive.update({k: False for k in loss_items})
            final_scores.append(weighted_loss_score(config, loss_items))
        if metric_outputs is not None:
            metric_values.update(metric_outputs.metric_values)
            is_positive.update(metric_outputs.is_positive)
            final_scores.append(metric_outputs.final_score)
        final_score = sum(final_scores) / len(final_scores)
        return MetricsOutputs(final_score, metric_values, is_positive)


class EnsembleFn(Protocol):
    def __call__(self, key: str, tensors: List[Tensor]) -> Tensor:
        pass


class DLEnsembleModel(CommonDLModel):
    ensemble_fn: Optional[EnsembleFn]

    def __init__(self, m: IDLModel, num_repeat: int) -> None:
        super().__init__()
        self.m = get_clones(m.m, num_repeat)
        self.ensemble_fn = None
        self.__identifier__ = m.__identifier__

    def build(self, config: DLConfig) -> None:
        raise ValueError("`build` should not be called for `DLEnsembleModel`")

    def forward(self, *args: Any, **kwargs: Any) -> forward_results_type:
        outputs: Dict[str, List[Tensor]] = {}
        for m in self.m:
            m_outputs = m(*args, **kwargs)
            if isinstance(m_outputs, Tensor):
                m_outputs = {PREDICTIONS_KEY: m_outputs}
            for k, v in m_outputs.items():
                outputs.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(outputs):
            if self.ensemble_fn is None:
                v = torch.stack(outputs[k]).mean(0)
            else:
                v = safe_execute(self.ensemble_fn, dict(key=k, tensors=outputs[k]))
            final_results[k] = v
        return final_results


__all__ = [
    "get_update_fn",
    "CommonTrainStep",
    "CommonDLModel",
    "DLEnsembleModel",
]
