import math
import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from torch.nn import Module
from torch.nn import ModuleList
from cftool.misc import context_error_handler

from ...types import tensor_dict_type
from ...types import tensor_tuple_type
from ...misc.toolkit import switch_requires_grad
from ...modules.blocks import MLP
from ...modules.blocks import CrossBase
from ...modules.blocks import CrossBlock
from ...modules.blocks import InvertibleBlock
from ...modules.blocks import MonotonousMapping
from ...modules.blocks import ConditionalBlocks
from ...modules.blocks import PseudoInvertibleBlock


def transition_builder(dim: int) -> Module:
    h_dim = int(dim // 2)
    return MonotonousMapping.make_couple(h_dim, h_dim, h_dim, "sigmoid", ascent=True)


def get_cond_mappings(
    condition_dim: int,
    cond_out_dim: Optional[int],
    num_units: List[int],
    to_latent: bool,
) -> ModuleList:
    cond_module = MLP.simple(
        condition_dim,
        cond_out_dim,
        num_units,
        bias=not to_latent,
        activation="glu",
    )
    return cond_module.mappings


responses_tuple_type = Tuple[List[Tensor], List[Tensor]]


class Pack(NamedTuple):
    net: Tensor
    cond: Tensor
    responses_tuple: responses_tuple_type


class MonoSplit(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ascent1: bool,
        ascent2: bool,
        to_latent: bool,
        builder: Callable[[int, int, bool], ConditionalBlocks],
    ) -> None:
        super().__init__()
        self.to_latent = to_latent
        self.m1 = builder(in_dim, out_dim, ascent1)
        self.m2 = builder(in_dim, out_dim, ascent2)

    def forward(
        self,
        net: Union[Tensor, tensor_tuple_type],
        cond: Union[Tensor, responses_tuple_type],
    ) -> Union[tensor_tuple_type, Pack]:
        if self.to_latent:
            assert isinstance(net, Tensor)
            return self.m1(net, cond), self.m2(net, cond)
        assert isinstance(net, tuple)
        if isinstance(cond, tuple):
            cond1, cond2 = cond
        else:
            cond1 = cond2 = cond
        o1, o2 = self.m1(net[0], cond1), self.m2(net[1], cond2)
        net = 0.5 * (o1.net + o2.net)
        cond = 0.5 * (o1.cond + o2.cond)
        return Pack(net, cond, (o1.responses, o2.responses))


class MonoCross(CrossBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        centralize: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        kwargs["bias"] = not centralize
        self.mapping = MonotonousMapping(in_dim, out_dim, ascent=True, **kwargs)
        self.centralize = centralize

    def forward(self, net: Tensor, cond: Tensor) -> Tensor:
        crossed = self.mapping(net * cond.abs()) * torch.sigmoid(cond)
        if self.centralize:
            return crossed
        return crossed + cond

    def extra_repr(self) -> str:
        return f"(centralize): {self.centralize}"

    @classmethod
    def make(
        cls,
        dim: int,
        inner: bool,
        *,
        centralize: bool = True,
        **kwargs: Any,
    ) -> "MonoCross":
        out_dim = 1 if inner else dim
        return cls(dim, out_dim, centralize=centralize, **kwargs)


def get_q_cross_builder(to_latent: bool) -> Callable[[int], Module]:
    return lambda dim: MonoCross.make(dim, to_latent, centralize=to_latent)


def get_y_cross_builder(to_latent: bool) -> Callable[[int], Module]:
    return lambda dim: MonoCross.make(dim, to_latent, centralize=not to_latent)


def monotonous_builder(
    is_q: bool,
    ascent1: bool,
    ascent2: bool,
    to_latent: bool,
    num_layers: int,
    condition_dim: int,
    cond_mappings: Optional[ModuleList] = None,
) -> Callable[[int, int], Module]:
    def _core(
        in_dim: int,
        out_dim: int,
        ascent: bool,
    ) -> ConditionalBlocks:
        cond_out_dim: Optional[int]
        block_out_dim: Optional[int]
        if to_latent:
            num_units = [out_dim] * (num_layers + 1)
            cond_out_dim = block_out_dim = None
        else:
            num_units = [in_dim] * num_layers
            # only cdf will initialize cond mappings here
            # q will utilize median mappings defined outside
            cond_out_dim = 1
            # block : y_mul / cdf_mul
            block_out_dim = 1

        blocks = MonotonousMapping.stack(
            in_dim,
            block_out_dim,
            num_units,
            ascent=ascent,
            use_couple_bias=(is_q and not to_latent) or (not is_q and to_latent),
            activation="sigmoid" if to_latent else "tanh",
            return_blocks=True,
        )
        assert isinstance(blocks, list)

        if cond_mappings is not None:
            cond_mappings_ = cond_mappings
        else:
            cond_mappings_ = get_cond_mappings(
                condition_dim,
                cond_out_dim,
                num_units,
                to_latent,
            )

        if is_q:
            cross_builder = get_q_cross_builder(to_latent)
        else:
            cross_builder = get_y_cross_builder(to_latent)
        cond_mixtures = ModuleList(
            [
                CrossBlock(
                    unit,
                    False,
                    cross_builder=cross_builder,
                )
                for unit in num_units
            ]
        )
        return ConditionalBlocks(
            ModuleList(blocks),
            cond_mappings_,
            add_last=to_latent,
            detach_condition=not to_latent,
            cond_mixtures=cond_mixtures,
        )

    def _split_core(in_dim: int, out_dim: int) -> Module:
        if not to_latent:
            in_dim = int(in_dim // 2)
        else:
            out_dim = int(out_dim // 2)
        return MonoSplit(in_dim, out_dim, ascent1, ascent2, to_latent, _core)

    return _split_core


class DDRCore(Module):
    def __init__(
        self,
        in_dim: int,
        fetch_q: bool,
        fetch_cdf: bool,
        num_layers: Optional[int] = None,
        num_blocks: Optional[int] = None,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        # common
        self.register_buffer("cdf_logit_anchor", torch.tensor([math.log(3.0)]))
        if not fetch_q and not fetch_cdf:
            raise ValueError("something must be fetched, either `q` or `cdf`")
        self.fetch_q = fetch_q
        self.fetch_cdf = fetch_cdf
        if num_layers is None:
            num_layers = 1
        if num_blocks is None:
            num_blocks = 2
        if num_blocks % 2 != 0:
            raise ValueError("`num_blocks` should be divided by 2")
        if latent_dim is None:
            latent_dim = 512
        self.latent_dim = latent_dim
        # median mappings
        median_units = [latent_dim] * num_layers
        self.median_mappings = get_cond_mappings(1, 3, median_units, False)
        # pseudo invertible q
        kwargs = {"num_layers": num_layers, "condition_dim": in_dim}
        if not self.fetch_q:
            q_to_latent_builder = self.dummy_builder
        else:
            q_to_latent_builder = monotonous_builder(
                is_q=True,
                ascent1=True,
                ascent2=True,
                to_latent=True,
                **kwargs,
            )
        if not self.fetch_cdf:
            q_from_latent_builder = self.dummy_builder
        else:
            q_from_latent_builder = monotonous_builder(
                is_q=True,
                ascent1=True,
                ascent2=True,
                to_latent=False,
                **kwargs,
            )
        self.q_invertible = PseudoInvertibleBlock(
            1,
            latent_dim,
            3,
            to_transition_builder=q_to_latent_builder,
            from_transition_builder=q_from_latent_builder,
        )
        # pseudo invertible y
        if not self.fetch_cdf:
            y_to_latent_builder = self.dummy_builder
        else:
            y_to_latent_builder = monotonous_builder(
                is_q=False,
                ascent1=True,
                ascent2=num_blocks == 0,
                to_latent=True,
                **kwargs,
            )
        if not self.fetch_q:
            y_from_latent_builder = self.dummy_builder
        else:
            y_from_latent_builder = monotonous_builder(
                is_q=False,
                ascent1=True,
                ascent2=True,
                to_latent=False,
                cond_mappings=self.median_mappings,
                **kwargs,
            )
        self.y_invertible = PseudoInvertibleBlock(
            1,
            latent_dim,
            3,
            to_transition_builder=y_to_latent_builder,
            from_transition_builder=y_from_latent_builder,
        )
        # q parameters
        if not self.fetch_q:
            self.q_parameters = []
        else:
            q_params1 = list(self.q_invertible.to_latent.parameters())
            q_params2 = list(self.y_invertible.from_latent.parameters())
            self.q_parameters = q_params1 + q_params2
        # invertible blocks
        self.num_blocks = num_blocks
        self.block_parameters: List[nn.Parameter] = []
        self.blocks = ModuleList()
        for _ in range(num_blocks):
            block = InvertibleBlock(latent_dim, transition_builder=transition_builder)
            self.block_parameters.extend(block.parameters())
            self.blocks.append(block)

    @property
    def q_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def q_inv_fn(self) -> Callable[[Tensor], Tensor]:
        return torch.sigmoid

    @property
    def dummy_builder(self) -> Callable[[int, int], Module]:
        return lambda _, __: nn.Identity()

    def _detach_q(self) -> context_error_handler:
        fetch_q = self.fetch_q

        def switch(requires_grad: bool) -> None:
            switch_requires_grad(self.q_parameters, requires_grad)
            switch_requires_grad(self.block_parameters, requires_grad)

        class _(context_error_handler):
            def __enter__(self) -> None:
                switch(not fetch_q)

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                switch(True)

        return _()

    @staticmethod
    def _get_median_outputs(pack: Pack) -> tensor_dict_type:
        cond_net = pack.cond
        med, pos_med_res, neg_med_res = cond_net.split(1, dim=1)
        return {"median": med, "pos_med_res": pos_med_res, "neg_med_res": neg_med_res}

    def _get_median_pack(self, net: Tensor) -> Pack:
        zeros = torch.zeros_like(net)
        res1, res2 = [], []
        for i, mapping in enumerate(self.median_mappings):
            net = mapping(net)
            if i == len(self.median_mappings) - 1:
                res1.append(zeros)
                res2.append(zeros)
            else:
                chunked = net.chunk(2, dim=1)
                res1.append(chunked[0])
                res2.append(chunked[1])
        return Pack(zeros, net, (res1, res2))

    @staticmethod
    def _merge_y_pack(
        pack: Pack,
        q_batch: Tensor,
        median_outputs: tensor_dict_type,
    ) -> tensor_dict_type:
        pos_med_res = median_outputs["pos_med_res"]
        neg_med_res = median_outputs["neg_med_res"]
        q_positive_mask = torch.sign(q_batch) == 1.0
        med_res = torch.where(q_positive_mask, pos_med_res, neg_med_res).detach()
        y_mul = pack.net
        y_res = med_res * y_mul
        return {
            "y_res": y_res,
            "med_mul": y_mul,
            "med_res": med_res,
            "q_positive_mask": q_positive_mask,
        }

    def _merge_q_pack(self, pack: Pack) -> tensor_dict_type:
        q_logit_mul = pack.net
        q_logit = self.cdf_logit_anchor * q_logit_mul
        return {
            "q": self.q_inv_fn(q_logit),
            "q_logit": q_logit,
            "q_logit_mul": q_logit_mul,
        }

    def _q_results(
        self,
        net: Tensor,
        q_batch: Tensor,
        median_outputs: tensor_dict_type,
        median_responses: responses_tuple_type,
        do_inverse: bool = False,
    ) -> tensor_dict_type:
        # prepare q_latent
        q_batch = self.q_fn(q_batch)
        q1, q2 = self.q_invertible(q_batch, net)
        q1, q2 = q1.net, q2.net
        # simulate quantile function
        for block in self.blocks:
            q1, q2 = block(q1, q2)
        y_pack = self.y_invertible.inverse((q1, q2), median_responses)
        assert isinstance(y_pack, Pack)
        results = self._merge_y_pack(y_pack, q_batch, median_outputs)
        q_inverse = None
        if do_inverse and self.fetch_cdf:
            y_res = results["y_res"]
            assert y_res is not None
            inverse_results = self._y_results(
                net,
                y_res.detach(),
                median_outputs,
                median_responses,
            )
            q_inverse = inverse_results["q"]
        results["q_inverse"] = q_inverse
        return results

    def _y_results(
        self,
        net: Tensor,
        y_batch: Tensor,
        median_outputs: tensor_dict_type,
        median_responses: responses_tuple_type,
        do_inverse: bool = False,
    ) -> tensor_dict_type:
        # prepare y_latent
        y1, y2 = self.y_invertible(y_batch, net)
        y1, y2 = y1.net, y2.net
        # simulate cdf
        with self._detach_q():
            for i in range(self.num_blocks):
                y1, y2 = self.blocks[self.num_blocks - i - 1].inverse(y1, y2)
            q_logit_pack = self.q_invertible.inverse((y1, y2), median_responses)
            assert isinstance(q_logit_pack, Pack)
            results = self._merge_q_pack(q_logit_pack)
            y_inverse_res = None
            if do_inverse and self.fetch_q:
                inverse_results = self._q_results(
                    net,
                    results["q"].detach(),
                    median_outputs,
                    median_responses,
                )
                y_inverse_res = inverse_results["y_res"]
            results["y_inverse_res"] = y_inverse_res
        return results

    def forward(
        self,
        net: Tensor,
        *,
        median: bool = False,
        do_inverse: bool = False,
        q_batch: Optional[Tensor] = None,
        y_batch: Optional[Tensor] = None,
    ) -> tensor_dict_type:
        median_pack = self._get_median_pack(net)
        median_responses = median_pack.responses_tuple
        results = median_outputs = self._get_median_outputs(median_pack)
        if self.fetch_q and not median and q_batch is not None:
            results.update(
                self._q_results(
                    net,
                    q_batch,
                    median_outputs,
                    median_responses,
                    do_inverse,
                )
            )
        if self.fetch_cdf and not median and y_batch is not None:
            results.update(
                self._y_results(
                    net,
                    y_batch,
                    median_outputs,
                    median_responses,
                    do_inverse,
                )
            )
        return results


__all__ = ["DDRCore"]
