from ..dl.pipeline import SimplePipeline as SimpleBase
from ..dl.pipeline import CarefreePipeline as CarefreeBase
from ...misc.internal_.inference import NLPInference


@SimpleBase.register("nlp.simple")
class SimplePipeline(SimpleBase):
    inference: NLPInference
    inference_base = NLPInference


@CarefreeBase.register("nlp.carefree")
class CarefreePipeline(CarefreeBase):
    inference: NLPInference
    inference_base = NLPInference


__all__ = [
    "SimplePipeline",
    "CarefreePipeline",
]
