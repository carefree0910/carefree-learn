from .api import *
from .data import *
from .models import *
from .trainer import *
from .pipeline import *
from .protocol import *
from .inference import *
from .configs import Configs
from .protocol import PrefetchLoader
from .misc._api import _remove
from .misc._api import _rmtree
from .modules.heads import HeadBase
from .modules.heads import HeadConfigs
from .modules.extractors import ExtractorBase
from .modules.aggregators import AggregatorBase
from .external_ import *

from pkg_resources import get_distribution

pkg = get_distribution("carefree-learn")
__version__ = pkg.version
