import torch.optim.lr_scheduler as scheduler

scheduler_dict = {}


def register_scheduler(name):
    def _register(cls_):
        global scheduler_dict
        scheduler_dict[name] = cls_
        return cls_

    return _register


register_scheduler("plateau")(scheduler.ReduceLROnPlateau)


@register_scheduler("warmup")
class WarmupScheduler(scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        multiplier,
        warmup_step,
        scheduler_afterwards_base=None,
        scheduler_afterwards_config=None,
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
    def lr_warmup_func(self):
        return lambda lr: lr * (
            (self.multiplier - 1.0) * self.last_epoch / self.warmup_step + 1.0
        )

    @property
    def lr_multiplier_func(self):
        return lambda lr: lr * self.multiplier

    @property
    def reduce_on_plateau_afterwards(self):
        return isinstance(self, scheduler.ReduceLROnPlateau)

    def _step_reduce_on_plateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch <= self.warmup_step:
            for param_group, lr in zip(
                self.optimizer.param_groups, map(self.lr_warmup_func, self.base_lrs)
            ):
                param_group["lr"] = lr
        else:
            if epoch is not None:
                epoch -= self.warmup_step
            self.scheduler_afterwards.step(metrics, None)

    def get_lr(self):
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

    def step(self, epoch=None, metrics=None):
        if self.reduce_on_plateau_afterwards:
            self._step_reduce_on_plateau(metrics, epoch)
        else:
            if not self.finished_warmup or self.scheduler_afterwards is None:
                return super().step(epoch)
            if epoch is not None:
                epoch -= self.warmup_step
            self.scheduler_afterwards.step(epoch)


__all__ = ["scheduler_dict", "register_scheduler"]
