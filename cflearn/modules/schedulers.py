import math

import numpy as np

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Callable
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..misc.toolkit import scheduler_requires_metric


class ISchedulerOp:
    @abstractmethod
    def schedule(self, step: int, **kwargs: Any) -> float:
        pass


scheduler_ops: Dict[str, Type[ISchedulerOp]] = {}
scheduler_dict: Dict[str, Type[_LRScheduler]] = {}


def register_scheduler(name: str) -> Callable:
    def _register(cls_: Type) -> Type:
        global scheduler_dict
        scheduler_dict[name] = cls_
        return cls_

    return _register


register_scheduler("cyclic")(CyclicLR)
register_scheduler("cosine")(CosineAnnealingLR)
register_scheduler("cosine_restarts")(CosineAnnealingWarmRestarts)


@register_scheduler("linear")
class LinearScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, *, start_epoch: int, end_epoch: int):
        def lr_lambda(epoch: int) -> float:
            span = float(end_epoch - start_epoch + 1)
            return 1.0 - max(0, min(epoch, end_epoch) + 1 - start_epoch) / span

        super().__init__(optimizer, lr_lambda=lr_lambda)


@register_scheduler("linear_inverse")
class LinearInverseScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, decay: float = 2.0e-5):
        self.decay = decay
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> List[float]:
        denom = 1.0 + self.decay * self._step_count
        return [base_lr / denom for base_lr in self.base_lrs]


@register_scheduler("step")
class StepLRWithFloor(StepLR):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        lr_floor: float = 1.0e-8,
    ):
        self.lr_floor = lr_floor
        super().__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        lrs = super().get_lr()
        return [max(lr, self.lr_floor) for lr in lrs]  # type: ignore

    def _get_closed_form_lr(self) -> List[float]:  # type: ignore
        lrs = super()._get_closed_form_lr()  # type: ignore
        return [max(lr, self.lr_floor) for lr in lrs]


@register_scheduler("exponential")
class ExponentialLRWithFloor(ExponentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
        lr_floor: float = 1.0e-8,
    ):
        self.lr_floor = lr_floor
        super().__init__(optimizer, gamma, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        lrs = super().get_lr()
        return [max(lr, self.lr_floor) for lr in lrs]  # type: ignore

    def _get_closed_form_lr(self) -> List[float]:  # type: ignore
        lrs = super()._get_closed_form_lr()  # type: ignore
        return [max(lr, self.lr_floor) for lr in lrs]


@register_scheduler("plateau")
class ReduceLROnPlateauWithGet(ReduceLROnPlateau):
    def get_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]  # type: ignore

    def get_last_lr(self) -> List[float]:
        return self.get_lr()


@register_scheduler("warmup")
class WarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: float,
        warmup_step: int,
        scheduler_afterwards_base: Type[_LRScheduler],
        scheduler_afterwards_config: Optional[Dict[str, Any]] = None,
    ):
        self.multiplier = multiplier
        assert self.multiplier > 1.0, "multiplier should be greater than 1"
        self.warmup_step, self.finished_warmup = warmup_step, False
        if scheduler_afterwards_config is None:
            scheduler_afterwards_config = {}
        self.scheduler_afterwards = scheduler_afterwards_base(
            optimizer,
            **scheduler_afterwards_config,
        )
        self.requires_metric = scheduler_requires_metric(self.scheduler_afterwards)
        super().__init__(optimizer)

    @property
    def lr_warmup_func(self) -> Callable[[float], float]:
        multiplier = (self.multiplier - 1.0) * self.last_epoch / self.warmup_step + 1.0  # type: ignore
        return lambda lr: lr * multiplier

    @property
    def lr_multiplier_func(self) -> Callable[[float], float]:
        return lambda lr: lr * self.multiplier

    def get_lr(self) -> List[float]:  # type: ignore
        if self.last_epoch > self.warmup_step:  # type: ignore
            if self.scheduler_afterwards is not None:
                if not self.finished_warmup:
                    self.finished_warmup = True
                    base_lrs = list(
                        map(self.lr_multiplier_func, self.base_lrs)  # type: ignore
                    )
                    self.scheduler_afterwards.base_lrs = base_lrs  # type: ignore
                return self.scheduler_afterwards.get_lr()  # type: ignore
            return list(map(self.lr_multiplier_func, self.base_lrs))
        return list(map(self.lr_warmup_func, self.base_lrs))  # type: ignore

    def get_last_lr(self) -> List[float]:
        if not self.finished_warmup:
            return super().get_last_lr()  # type: ignore
        return self.scheduler_afterwards.get_last_lr()  # type: ignore

    def step(self, metrics: Optional[float] = None) -> None:
        if not self.finished_warmup or self.scheduler_afterwards is None:
            return super().step()
        if not self.requires_metric:
            self.scheduler_afterwards.step()
        else:
            assert metrics is not None
            self.scheduler_afterwards.step(metrics)  # type: ignore


def register_op(name: str) -> Callable:
    def _register(cls_: Type) -> Type:
        global scheduler_ops
        scheduler_ops[name] = cls_
        return cls_

    return _register


@register_op("cosine_warmup")
class CosineWarmupOp:
    def __init__(
        self,
        warmup_steps: List[int],
        cycle_lengths: List[int],
        f_start: List[float],
        f_min: List[float],
        f_max: List[float],
    ):
        self.warmup_steps = warmup_steps
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max

    def find_in_interval(self, step: int) -> int:
        interval = 0
        for cl in self.cum_cycles[1:]:
            if step <= cl:
                return interval
            interval += 1
        return interval - 1

    def schedule(self, step: int, **kwargs: Any) -> float:
        cycle = self.find_in_interval(step)
        step = step - self.cum_cycles[cycle]
        warmup_step = self.warmup_steps[cycle]
        f_start = self.f_start[cycle]
        f_max = self.f_max[cycle]
        if step < warmup_step:
            return (f_max - f_start) / warmup_step * step + f_start
        t = (step - warmup_step) / (self.cycle_lengths[cycle] - warmup_step)
        t = min(t, 1.0)
        f_min = self.f_min[cycle]
        return f_min + 0.5 * (f_max - f_min) * (1.0 + math.cos(t * math.pi))


@register_op("linear_warmup")
class LinearWarmupOp(CosineWarmupOp):
    def schedule(self, step: int, **kwargs: Any) -> float:
        cycle = self.find_in_interval(step)
        step = step - self.cum_cycles[cycle]
        warmup_step = self.warmup_steps[cycle]
        f_start = self.f_start[cycle]
        f_max = self.f_max[cycle]
        if step < warmup_step:
            return (f_max - f_start) / warmup_step * step + f_start
        cycle_length = self.cycle_lengths[cycle]
        f_min = self.f_min[cycle]
        return f_min + (f_max - f_min) * (cycle_length - step) / cycle_length


@register_scheduler("op")
class OpScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, op_type: str, op_config: Dict[str, Any]):
        op_base = scheduler_ops.get(op_type)
        if op_base is None:
            raise ValueError(f"unrecognized scheduler op '{op_type}' occurred")
        op = op_base(**op_config)
        super().__init__(optimizer, lr_lambda=op.schedule)


__all__ = [
    "scheduler_dict",
    "register_scheduler",
    "LinearInverseScheduler",
    "StepLRWithFloor",
    "ExponentialLRWithFloor",
    "ReduceLROnPlateauWithGet",
    "WarmupScheduler",
]
