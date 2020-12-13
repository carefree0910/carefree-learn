from typing import *
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..misc.toolkit import scheduler_requires_metric

scheduler_dict = {}


def register_scheduler(name: str) -> Callable[[Type], Type]:
    def _register(cls_: Type) -> Type:
        global scheduler_dict
        scheduler_dict[name] = cls_
        return cls_

    return _register


register_scheduler("cyclic")(CyclicLR)
register_scheduler("cosine")(CosineAnnealingLR)
register_scheduler("cosine_restarts")(CosineAnnealingWarmRestarts)


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
            optimizer, **scheduler_afterwards_config
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


__all__ = ["scheduler_dict", "register_scheduler"]
