from typing import Any
from typing import Optional

from ...data import MLLoader
from ...schema import ILoss
from ...schema import TrainerState
from ...schema import _IMetric
from ...schema import InferenceOutputs
from ...schema import IInference


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
