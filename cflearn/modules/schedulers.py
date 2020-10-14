from typing import *
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

scheduler_dict = {}


def register_scheduler(name: str) -> Callable[[Type], Type]:
    def _register(cls_: Type) -> Type:
        global scheduler_dict
        scheduler_dict[name] = cls_
        return cls_

    return _register


register_scheduler("plateau")(ReduceLROnPlateau)


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
        super().__init__(optimizer)

    @property
    def lr_warmup_func(self) -> Callable[[float], float]:
        multiplier = (self.multiplier - 1.0) * self.last_epoch / self.warmup_step + 1.0
        return lambda lr: lr * multiplier

    @property
    def lr_multiplier_func(self) -> Callable[[float], float]:
        return lambda lr: lr * self.multiplier

    @property
    def reduce_on_plateau_afterwards(self) -> bool:
        return isinstance(self, ReduceLROnPlateau)

    def _step_reduce_on_plateau(
        self,
        metrics: float,
        epoch: Optional[int] = None,
    ) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch: int = epoch
        if self.last_epoch <= self.warmup_step:
            for param_group, lr in zip(
                self.optimizer.param_groups, map(self.lr_warmup_func, self.base_lrs)
            ):
                param_group["lr"] = lr
        else:
            if epoch is not None:
                epoch -= self.warmup_step
            self.scheduler_afterwards.step(metrics, None)

    def get_lr(self) -> Union[float, List[float]]:
        if self.last_epoch > self.warmup_step:
            if self.scheduler_afterwards is not None:
                if not self.finished_warmup:
                    self.finished_warmup = True
                    self.scheduler_afterwards.base_lrs = list(
                        map(self.lr_multiplier_func, self.base_lrs)
                    )
                return self.scheduler_afterwards.get_lr()
            return list(map(self.lr_multiplier_func, self.base_lrs))
        return list(map(self.lr_warmup_func, self.base_lrs))

    def step(
        self,
        epoch: Optional[int] = None,
        metrics: Optional[float] = None,
    ) -> None:
        if self.reduce_on_plateau_afterwards:
            assert metrics is not None
            self._step_reduce_on_plateau(metrics, epoch)
        else:
            if not self.finished_warmup or self.scheduler_afterwards is None:
                return super().step(epoch)
            if epoch is not None:
                epoch -= self.warmup_step
            self.scheduler_afterwards.step(epoch)


__all__ = ["scheduler_dict", "register_scheduler"]
