from typing import Any
from typing import Dict
from typing import Tuple

from ....schema import IRuntimeDataBlock
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import ORIGINAL_LABEL_KEY


@IRuntimeDataBlock.register("tuple_to_batch")
class TupleToBatchBlock(IRuntimeDataBlock):
    def postprocess_item(self, item: Tuple[Any, Any]) -> Dict[str, Any]:
        image, labels = item
        return {
            INPUT_KEY: image,
            LABEL_KEY: labels,
            ORIGINAL_LABEL_KEY: labels,
        }


__all__ = ["TupleToBatchBlock"]
