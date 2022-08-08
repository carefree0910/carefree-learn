from ...pipeline import DLPipeline
from ...misc.internal_.inference import CVInference


@DLPipeline.register("cv")
class CVPipeline(DLPipeline):
    inference: CVInference
    inference_base = CVInference


__all__ = [
    "CVPipeline",
]
