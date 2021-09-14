from typing import Type

from ..dl.pipeline import SimplePipeline as SimpleBase
from ..dl.pipeline import CarefreePipeline as CarefreeBase
from ...misc.internal_ import CVInference


@SimpleBase.register("cv.simple")
class SimplePipeline(SimpleBase):
    inference: CVInference
    inference_base = Type[CVInference]


@CarefreeBase.register("cv.carefree")
class CarefreePipeline(CarefreeBase):
    inference: CVInference
    inference_base = Type[CVInference]


__all__ = [
    "SimplePipeline",
    "CarefreePipeline",
]
