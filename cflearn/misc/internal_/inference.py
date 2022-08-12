from typing import Any
from typing import Optional

from ...data import MLLoader
from ...protocol import ILoss
from ...protocol import TrainerState
from ...protocol import _IMetric
from ...protocol import InferenceOutputs
from ...protocol import IInference


class MLInference(IInference):
    def get_outputs(  # type: ignore
        self,
        loader: MLLoader,
        *,
        portion: float = 1.0,
        use_loader_cache: bool = True,
        state: Optional[TrainerState] = None,
        metrics: Optional[_IMetric] = None,
        loss: Optional[ILoss] = None,
        return_outputs: bool = True,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> InferenceOutputs:
        if use_loader_cache:
            kwargs["loader_name"] = loader.name
        return super().get_outputs(
            loader,
            portion=portion,
            state=state,
            metrics=metrics,
            loss=loss,
            return_outputs=return_outputs,
            use_tqdm=use_tqdm,
            **kwargs,
        )


class CVInference(IInference):
    pass


class NLPInference(IInference):
    pass


__all__ = [
    "MLInference",
    "CVInference",
    "NLPInference",
]
