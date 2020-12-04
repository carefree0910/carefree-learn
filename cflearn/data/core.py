from cfdata.tabular import DataLoader
from cfdata.tabular import ImbalancedSampler
from cfdata.tabular import TabularData as TD

from ..protocol import DataProtocol
from ..protocol import SamplerProtocol
from ..protocol import DataLoaderProtocol


@DataProtocol.register("tabular")
class TabularData(TD, DataProtocol):
    pass


@SamplerProtocol.register("tabular")
class TabularSampler(ImbalancedSampler, SamplerProtocol):
    pass


@DataLoaderProtocol.register("tabular")
class TabularLoader(DataLoader, DataLoaderProtocol):
    pass


__all__ = [
    "TabularData",
    "TabularSampler",
    "TabularLoader",
]
