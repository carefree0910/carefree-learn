from ...pipeline import DLPipeline
from ...misc.internal_.inference import NLPInference


@DLPipeline.register("nlp")
class NLPPipeline(DLPipeline):
    inference: NLPInference
    inference_base = NLPInference


__all__ = [
    "NLPPipeline",
]
