from ...pipeline import DLPipeline
from ...implementations.inference import NLPInference


@DLPipeline.register("nlp")
class NLPPipeline(DLPipeline):
    inference: NLPInference
    inference_base = NLPInference


__all__ = [
    "NLPPipeline",
]
