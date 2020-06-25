import os

from typing import *
from cftool.misc import *
from cftool.ml.utils import *
from cftool.ml.param_utils import *
from cfdata.tabular import *
from functools import partial
from cftool.ml.hpo import HPOBase

from .dist import *
from .bases import *
from .models import *
from .modules import *
from .misc.toolkit import eval_context, Initializer


# register

def register_initializer(name):
    def _register(f):
        Initializer.add_initializer(f, name)
        return f
    return _register
