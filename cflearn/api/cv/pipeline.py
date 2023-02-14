from typing import Any
from typing import Dict
from typing import Optional

from ...pipeline import IModifier
from ...pipeline import DLPipeline
from ...data.cv import Transforms
from ...implementations.inference import CVInference


class ICVPipeline:
    test_transform: Optional[Transforms]


@IModifier.register("cv")
class CVModifier(IModifier, ICVPipeline):
    build_steps = ["setup_test_transform"] + IModifier.build_steps

    # build steps

    def setup_test_transform(self, data_info: Dict[str, Any]) -> None:
        # at loading stage, `"test_transform"` will be injected to `data_info`
        self.test_transform = data_info.get("test_transform")


@DLPipeline.register("cv")
class CVPipeline(ICVPipeline, DLPipeline):
    modifier = "cv"
    inference: CVInference
    inference_base = CVInference


__all__ = [
    "CVPipeline",
]
