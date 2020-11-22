# Contributing

Thank you for your interest in contributing to `carefree-learn`! Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    - Post about your intended feature in an [issue](https://github.com/carefree0910/carefree-learn/issues), and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue in the [`carefree-learn` issue list](https://github.com/carefree0910/carefree-learn/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please include some unittests and submit a Pull Request to https://github.com/carefree0910/carefree-learn.


## Developing

To develop `carefree-learn` on your machine, here are some tips:

1. Uninstall all existing `carefree-learn` installs:
```bash
conda uninstall carefree-learn
pip uninstall carefree-learn
```

1. Follow [Installation Guide](https://carefree0910.me/carefree-learn-doc/docs/getting-started/installation) to install `carefree-learn`. Remember to choose the `GitHub` tab in the [pip installation](https://carefree0910.me/carefree-learn-doc/docs/getting-started/installation#pip-installation) section.

2. Follow [Style Guide](#style-guide) and happy coding!


## Style Guide

`carefree-learn` adopted [`black`](https://github.com/psf/black) and [`mypy`](https://github.com/python/mypy) to stylize its codes, so you may need to check the format, coding style and type hint with them before your codes could actually be merged.

Besides, there are a few more principles that I'm using for sorting imports:
+ From short to long (for both naming and path).
+ From *a* to *z* (alphabetically).
+ Divided into four sections:
  1. `import ...`
  2. `import ... as ...`
  3. `from ... import ...`
  4. relative imports
+ From general to specific (a `*` will always appear at the top of each section)

Here's an example to illustrate these ([source code](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/api/auto.py)):

```python
import os
import json
import torch
import optuna

import numpy as np
import optuna.visualization as vis

from typing import *
from functools import partial
from tqdm.autonotebook import tqdm
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.ml.utils import scoring_fn_type
from cfdata.tabular import task_type_type
from cfdata.tabular import parse_task_type
from cfdata.tabular import TabularData
from optuna.trial import TrialState
from optuna.trial import FrozenTrial
from optuna.importance import BaseImportanceEvaluator
from plotly.graph_objects import Figure

from .basic import *
from .ensemble import *
from .hpo import optuna_tune
from .hpo import default_scoring
from .hpo import optuna_params_type
from .hpo import OptunaPresetParams
from .production import Pack
from .production import Predictor
from ..types import data_type
```

But after all, this is not a strict constraint so everything will be fine as long as it 'looks good'🤣


## Components

In `carefree-learn`, we've divided implementations into two parts: the [`Modules`](#modules) part and the [`Models`](#models) part. Basically, we construct various [`Modules`](#modules) to form our final [`Models`](#models).

### Modules

For modules, `carefree-learn` continue to divide them into two sections: the basic [`Blocks`](#blocks) and [`pipe`](#pipe) related ones. Basically, a [`pipe`](#pipe) is constructed by `transform`, `extractor` and `head`, and each of them may utilize one of the [`Blocks`](#blocks).

#### Blocks

+ Please refer to [Common Blocks](https://carefree0910.me/carefree-learn-doc/docs/design-principles#common-blocks) for detailed design principles.

#### `pipe`

+ Please refer to [`pipe`](https://carefree0910.me/carefree-learn-doc/docs/design-principles#pipe) for detailed design principles.
+ Please refer to [`Customizing New Modules`](https://carefree0910.me/carefree-learn-doc/docs/developer-guides/customization#customizing-new-modules) for how to implement new `extractor` and new `head`.

:::tip
It is recommended to follow the implementation style in `carefree-learn` to make new implementations:
+ For `extractor`, please refer to the [rnn](https://github.com/carefree0910/carefree-learn/tree/dev/cflearn/modules/extractors/rnn) implementation.
+ For `head`, please refer to the [fcnn](https://github.com/carefree0910/carefree-learn/tree/dev/cflearn/modules/heads/fcnn) implementation.
:::

### Models

For models, `carefree-learn` will leverage existing [`Modules`](#modules) to construct them. Please refer to [Constructing Existing Modules](https://carefree0910.me/carefree-learn-doc/docs/developer-guides/customization#constructing-existing-modules) to see what's going under the hood and how could we implement new models.


## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

### Title

The title of your pull request should

+ briefly describe and reflect the changes
+ wrap any code with backticks

### Description

The description of your pull request should

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks
