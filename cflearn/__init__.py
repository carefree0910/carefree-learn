from .schema import *
from .constants import *
from .parameters import *

from .data import *
from .modules import *
from .zoo import *

from .api import *

from . import scripts

from pkg_resources import get_distribution

pkg = get_distribution("carefree-learn")
__version__ = pkg.version
