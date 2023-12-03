import math

from typing import Optional
from cftool.misc import Incrementer

from .schema import TrainerMonitor


@TrainerMonitor.register("basic")
class BasicMonitor(TrainerMonitor):
    def __init__(self, patience: int = 25):  # type: ignore
        super().__init__()
        self.patience = patience
        self.num_snapshot = 0
        self.best_score = -math.inf
        self.worst_score: Optional[float] = None

    def snapshot(self, new_score: float) -> bool:
        self.num_snapshot += 1
        if self.worst_score is None:
            self.worst_score = new_score
        else:
            self.worst_score = min(new_score, self.worst_score)
        if new_score > self.best_score:
            self.best_score = new_score
            return True
        return False

    def check_terminate(self, new_score: float) -> bool:
        if self.num_snapshot <= self.patience:
            return False
        if self.worst_score is None:
            return False
        return new_score <= self.worst_score

    def punish_extension(self) -> None:
        return None


@TrainerMonitor.register("mean_std")
class MeanStdMonitor(BasicMonitor):
    def __init__(
        self,
        *,
        patience: int = 5,
        window_size: int = 25,
        overfit_tolerance: float = 25.0,
    ):
        super().__init__()
        self.patience = patience
        self.overfit_tolerance = overfit_tolerance
        self.best_score = -math.inf
        self.overfit_level = 0.0
        self._incrementer = Incrementer(window_size)

    def snapshot(self, new_score: float) -> bool:
        self._incrementer.update(new_score)
        mean, std = self._incrementer.mean, self._incrementer.std
        std = max(std, 1.0e-8)
        if new_score < mean - std:
            max_decrease = self.overfit_tolerance / self.patience
            decrease = min(max_decrease, (mean - new_score) / std + 1.0)
            self.overfit_level += decrease
        elif new_score > mean + std:
            improvement = (new_score - mean) / std - 1.0
            self.overfit_level = max(0.0, self.overfit_level - improvement)
        return super().snapshot(new_score)

    def check_terminate(self, new_score: float) -> bool:
        if self.num_snapshot <= 10:
            return False
        if self.overfit_level >= self.overfit_tolerance:
            return True
        return False


@TrainerMonitor.register("plateau")
class PlateauMonitor(TrainerMonitor):
    def __init__(
        self,
        *,
        patience: float = 5.0,
        window_size: int = 25,
        plateau_tolerance: float = 25.0,
        plateau_threshold: float = 0.2,
    ):
        super().__init__()
        self.patience = patience
        self.window_size = window_size
        self.plateau_tolerance = plateau_tolerance
        self.plateau_threshold = plateau_threshold
        self.num_snapshot = 0
        self.plateau_level = 0.0
        self._incrementer = Incrementer(window_size)

    @property
    def max_plateau_increase(self) -> float:
        return self.plateau_tolerance / self.patience

    def snapshot(self, new_score: float) -> bool:
        self.num_snapshot += 1
        self._incrementer.update(new_score)
        if self.num_snapshot <= self.window_size:
            return False
        mean, std = self._incrementer.mean, self._incrementer.std
        ratio = max(abs(new_score - mean) / max(std, 1.0e-8), 1.0e-8)
        if ratio < self.plateau_threshold:
            plateau = min(
                self.max_plateau_increase,
                1.0 / ratio - 1.0 / self.plateau_threshold,
            )
            self.plateau_level += plateau
        return False

    def check_terminate(self, new_score: float) -> bool:
        if self.plateau_level >= self.plateau_tolerance:
            return True
        return False

    def punish_extension(self) -> None:
        self.plateau_level += self.max_plateau_increase / 5.0


@TrainerMonitor.register("conservative")
class ConservativeMonitor(TrainerMonitor):
    def snapshot(self, new_score: float) -> bool:
        return True

    def check_terminate(self, new_score: float) -> bool:
        return False

    def punish_extension(self) -> None:
        pass


@TrainerMonitor.register("lazy")
class LazyMonitor(TrainerMonitor):
    def snapshot(self, new_score: float) -> bool:
        return False

    def check_terminate(self, new_score: float) -> bool:
        return False

    def punish_extension(self) -> None:
        pass


__all__ = [
    "BasicMonitor",
    "MeanStdMonitor",
    "PlateauMonitor",
    "ConservativeMonitor",
    "LazyMonitor",
]
