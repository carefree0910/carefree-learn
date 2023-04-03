from typing import List
from typing import Type
from typing import Optional
from torchvision.datasets import MNIST

from .api import TorchData
from .api import TorchDataConfig
from ..blocks import HWCToCHWBlock
from ..blocks import ToNumpyBlock
from ..blocks import TupleToBatchBlock
from ..blocks import StaticNormalizeBlock
from ...schema import IDataBlock
from ...schema import DataProcessorConfig
from ...parameters import OPT


def mnist_data(
    config: Optional[TorchDataConfig] = None,
    processor_config: Optional[DataProcessorConfig] = None,
    *,
    cache_root: str = OPT.data_cache_dir,
    additional_blocks: Optional[List[IDataBlock]] = None,
) -> TorchData:
    if processor_config is None:
        processor_config = DataProcessorConfig()
    if processor_config.block_names is None:
        processor_config.set_blocks(
            TupleToBatchBlock(),
            ToNumpyBlock(),
            StaticNormalizeBlock(),
            HWCToCHWBlock(),
        )
    if additional_blocks is not None:
        processor_config.add_blocks(*additional_blocks)
    train_data = MNIST(cache_root, download=True)
    valid_data = MNIST(cache_root, train=False, download=True)
    return TorchData.build(
        train_data,
        valid_data,
        config=config,
        processor_config=processor_config,
    )


__all__ = [
    "mnist_data",
]
