import torch
import cflearn

import numpy as np

from typing import Any
from typing import Dict
from cflearn.modules.blocks import Linear

# for reproduction
np.random.seed(142857)
torch.manual_seed(142857)

# prepare
dim = 5
num_data = 10000

x = np.random.random([num_data, dim]) * 2.0
y_add = np.sum(x, axis=1, keepdims=True)
y_prod = np.prod(x, axis=1, keepdims=True)
y_mix = np.hstack([y_add, y_prod])

kwargs = {"task_type": "reg", "use_simplify_data": True}

# add
linear = cflearn.make("linear", **kwargs).fit(x, y_add)
fcnn = cflearn.make("fcnn", **kwargs).fit(x, y_add)
cflearn.evaluate(x, y_add, pipelines=[linear, fcnn])

linear_core = linear.model.heads["linear"].linear
print(f"w: {linear_core.weight.data}, b: {linear_core.bias.data}")

# prod


@cflearn.register_extractor("prod_extractor")
class ProdExtractor(cflearn.ExtractorBase):
    @property
    def out_dim(self) -> int:
        return 1

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return net.prod(dim=1, keepdim=True)


cflearn.register_config("prod_extractor", "default", config={})


@cflearn.register_model("prod")
@cflearn.register_pipe("linear", extractor="prod_extractor")
class ProdNetwork(cflearn.ModelBase):
    pass


linear = cflearn.make("linear", **kwargs).fit(x, y_prod)
fcnn = cflearn.make("fcnn", **kwargs).fit(x, y_prod)
prod = cflearn.make("prod", **kwargs).fit(x, y_prod)
cflearn.evaluate(x, y_prod, pipelines=[linear, fcnn, prod])

prod_linear = prod.model.heads["linear"].linear
print(f"w: {prod_linear.weight.item():8.6f}, b: {prod_linear.bias.item():8.6f}")

# mixture


@cflearn.register_head("mixture")
class MixtureHead(cflearn.HeadBase):
    def __init__(self, in_dim: int, out_dim: int, target_dim: int):
        super().__init__(in_dim, out_dim)
        self.dim = target_dim
        self.linear = Linear(in_dim, 1)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        target = self.linear(net)
        zeros = torch.zeros_like(target)
        tensors = [target, zeros] if self.dim == 0 else [zeros, target]
        return torch.cat(tensors, dim=1)


cflearn.register_head_config("mixture", "add", head_config={"target_dim": 0})
cflearn.register_head_config("mixture", "prod", head_config={"target_dim": 1})


@cflearn.register_model("mixture")
@cflearn.register_pipe(
    "add",
    extractor="identity",
    head="mixture",
    head_config="add",
)
@cflearn.register_pipe(
    "prod",
    extractor="prod_extractor",
    head="mixture",
    head_config="prod",
)
class MixtureNetwork(cflearn.ModelBase):
    pass


linear = cflearn.make("linear", **kwargs).fit(x, y_mix)
fcnn = cflearn.make("fcnn", **kwargs).fit(x, y_mix)
prod = cflearn.make("prod", **kwargs).fit(x, y_mix)
mixture = cflearn.make("mixture", **kwargs).fit(x, y_mix)
cflearn.evaluate(x, y_mix, pipelines=[linear, fcnn, prod, mixture])

add_linear = mixture.model.heads["add"].linear
prod_linear = mixture.model.heads["prod"].linear
print(f"add  w: {add_linear.weight.data}")
print(f"add  b: {add_linear.bias.data}")
print(f"prod w: {prod_linear.weight.data}")
print(f"prod b: {prod_linear.bias.data}")
