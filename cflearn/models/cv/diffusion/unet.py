import math
import torch

from abc import abstractmethod
from abc import ABCMeta
from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

from ....modules.blocks import conv_nd
from ....modules.blocks import zero_module
from ....modules.blocks import HijackLinear
from ....modules.blocks import SpatialTransformer
from ....modules.blocks import MultiHeadSpatialAttention
from ....modules.blocks import ResidualBlockWithTimeEmbedding
from ....modules.blocks.convs.residual import ResUpsample
from ....modules.blocks.convs.residual import ResDownsample


class TimestepBlock(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, net: Tensor, time_net: Tensor) -> Tensor:
        pass


class TimestepAttnSequential(TimestepBlock, nn.Sequential, metaclass=ABCMeta):
    def forward(
        self,
        net: Tensor,
        time_net: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                net = layer(net, time_net)
            elif isinstance(layer, SpatialTransformer):
                net = layer(net, context)
            else:
                net = layer(net)
        return net


class ResBlock(ResidualBlockWithTimeEmbedding, TimestepBlock):
    pass


def timestep_embedding(
    # 1-D Tensor, shape=[B]
    timesteps: Tensor,
    output_dim: int,
    *,
    dtype: torch.dtype,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> Tensor:
    if repeat_only:
        return timesteps[..., None].repeat_interleave(output_dim, dim=1)
    half = output_dim // 2
    frequency = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * frequency[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if output_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    embedding = embedding.to(dtype)
    return embedding


class UNetDiffuser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = None,
        use_spatial_transformer: bool = False,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        start_channels: int = 320,
        num_res_blocks: int = 2,
        attention_downsample_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        resample_with_conv: bool = True,
        resample_with_resblock: bool = False,
        use_scale_shift_norm: bool = False,
        num_classes: Optional[int] = None,
        use_linear_in_transformer: bool = False,
        # misc
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        tome_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.signal_dim = signal_dim
        self.start_channels = start_channels
        self.num_res_blocks = num_res_blocks
        self.attention_downsample_rates = attention_downsample_rates
        self.dropout = dropout
        self.channel_multipliers = channel_multipliers
        self.resample_with_conv = resample_with_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_classes = num_classes
        self.use_linear_in_transformer = use_linear_in_transformer
        self.use_checkpoint = use_checkpoint
        self.attn_split_chunk = attn_split_chunk

        time_embedding_dim = start_channels * 4
        self.time_embedding = nn.Sequential(
            HijackLinear(start_channels, time_embedding_dim),
            nn.SiLU(),
            HijackLinear(time_embedding_dim, time_embedding_dim),
        )

        if num_classes is None:
            self.label_embedding = None
        else:
            self.label_embedding = nn.Embedding(num_classes, time_embedding_dim)

        make_res_block = lambda in_c, out_c, **kwargs: ResBlock(
            in_c,
            out_c,
            signal_dim=signal_dim,
            norm_eps=1.0e-5,
            time_embedding_channels=time_embedding_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            **kwargs,
        )

        def make_attn_block(in_c: int) -> nn.Module:
            err_msg = "either `num_heads` or `num_head_channels` should be provided"
            if num_head_channels is not None:
                n_heads = in_c // num_head_channels
            else:
                if num_heads is None:
                    raise ValueError(err_msg)
                n_heads = num_heads
            head_c = in_c // n_heads if num_head_channels is None else num_head_channels
            if not use_spatial_transformer:
                # TODO: support attn_split_chunk
                return MultiHeadSpatialAttention(
                    in_c,
                    num_heads=n_heads,
                    num_head_channels=head_c,
                    use_checkpoint=use_checkpoint,
                )
            return SpatialTransformer(
                in_c,
                n_heads,
                head_c,
                num_layers=num_transformer_layers,
                context_dim=context_dim,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                attn_split_chunk=attn_split_chunk,
                tome_info=tome_info,
            )

        def make_downsample(in_c: int, out_c: int) -> TimestepAttnSequential:
            if not resample_with_resblock:
                return TimestepAttnSequential(
                    ResDownsample(
                        in_c,
                        resample_with_conv,
                        signal_dim=signal_dim,
                        out_channels=out_c,
                    )
                )
            res_block = make_res_block(in_c, out_c, integrate_downsample=True)
            return TimestepAttnSequential(res_block)

        def make_upsample(in_c: int, out_c: int) -> nn.Module:
            if not resample_with_resblock:
                return ResUpsample(
                    in_c,
                    resample_with_conv,
                    signal_dim=signal_dim,
                    out_channels=out_c,
                )
            return make_res_block(in_c, out_c, integrate_upsample=True)

        # input
        input_blocks = [
            TimestepAttnSequential(
                conv_nd(signal_dim, in_channels, start_channels, 3, padding=1)
            ),
        ]
        feature_size = start_channels
        input_block_channels = [start_channels]
        in_nc = start_channels
        downsample_rate = 1
        for i, multiplier in enumerate(channel_multipliers):
            for _ in range(num_res_blocks):
                out_nc = multiplier * start_channels
                blocks = [make_res_block(in_nc, out_nc)]
                in_nc = out_nc
                if downsample_rate in attention_downsample_rates:
                    blocks.append(make_attn_block(in_nc))
                input_blocks.append(TimestepAttnSequential(*blocks))
                feature_size += in_nc
                input_block_channels.append(in_nc)
            if i != len(channel_multipliers) - 1:
                out_nc = in_nc
                input_blocks.append(make_downsample(in_nc, out_nc))
                downsample_rate *= 2
                feature_size += in_nc
                input_block_channels.append(in_nc)
        self.input_blocks = nn.ModuleList(input_blocks)

        # residual
        self.residual = TimestepAttnSequential(
            make_res_block(in_nc, in_nc),
            make_attn_block(in_nc),
            make_res_block(in_nc, in_nc),
        )
        feature_size += in_nc

        # output
        output_blocks = []
        for i, multiplier in list(enumerate(channel_multipliers))[::-1]:
            for idx in range(num_res_blocks + 1):
                idx_nc = input_block_channels.pop()
                out_nc = start_channels * multiplier
                blocks = [make_res_block(in_nc + idx_nc, out_nc)]
                in_nc = out_nc
                if downsample_rate in attention_downsample_rates:
                    blocks.append(make_attn_block(in_nc))
                if i != 0 and idx == num_res_blocks:
                    out_nc = in_nc
                    blocks.append(make_upsample(in_nc, out_nc))
                    downsample_rate //= 2
                output_blocks.append(TimestepAttnSequential(*blocks))
                feature_size += in_nc
        self.output_blocks = nn.ModuleList(output_blocks)

        # head
        head_conv = conv_nd(signal_dim, start_channels, out_channels, 3, padding=1)
        self.head = nn.Sequential(
            nn.GroupNorm(32, in_nc),
            nn.SiLU(),
            zero_module(head_conv),
        )

    def set_tome_info(self, tome_info: Optional[Dict[str, Any]]) -> None:
        for m in self.modules():
            if isinstance(m, SpatialTransformer):
                m.set_tome_info(tome_info)

    def forward(
        self,
        net: Tensor,
        *,
        timesteps: Tensor,
        context: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        control: Optional[List[Tensor]] = None,
        only_mid_control: bool = False,
    ) -> Tensor:
        if (labels is None) ^ (self.num_classes is None):
            raise ValueError("`labels` should be given iff `num_classes` is specified")

        # tomesd
        for m in self.modules():
            if isinstance(m, SpatialTransformer):
                for block in m.blocks:
                    if block.tome_info is not None:
                        block.tome_info["size"] = net.shape[-2:]

        # timenet
        time_net = timestep_embedding(
            timesteps,
            self.start_channels,
            dtype=net.dtype,
            repeat_only=False,
        )
        time_net = self.time_embedding(time_net)
        if self.label_embedding is not None:
            time_net = time_net + self.label_embedding(labels)

        # main
        nets = []
        for block in self.input_blocks:
            net = block(net, time_net, context)
            nets.append(net)
        net = self.residual(net, time_net, context)
        if control is not None:
            net += control.pop()
        for block in self.output_blocks:
            if only_mid_control or control is None:
                net = torch.cat([net, nets.pop()], dim=1)
            else:
                net = torch.cat([net, nets.pop() + control.pop()], dim=1)
            net = block(net, time_net, context)
        net = self.head(net)

        return net


class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hint_channels: int,
        *,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = None,
        use_spatial_transformer: bool = False,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        start_channels: int = 320,
        num_res_blocks: int = 2,
        attention_downsample_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        resample_with_conv: bool = True,
        resample_with_resblock: bool = False,
        use_scale_shift_norm: bool = False,
        num_classes: Optional[int] = None,
        use_linear_in_transformer: bool = False,
        # misc
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        tome_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hint_channels = hint_channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.signal_dim = signal_dim
        self.start_channels = start_channels
        self.num_res_blocks = num_res_blocks
        self.attention_downsample_rates = attention_downsample_rates
        self.dropout = dropout
        self.channel_multipliers = channel_multipliers
        self.resample_with_conv = resample_with_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_classes = num_classes
        self.use_linear_in_transformer = use_linear_in_transformer
        self.use_checkpoint = use_checkpoint
        self.attn_split_chunk = attn_split_chunk

        time_embedding_dim = start_channels * 4
        self.time_embed = nn.Sequential(
            HijackLinear(start_channels, time_embedding_dim),
            nn.SiLU(),
            HijackLinear(time_embedding_dim, time_embedding_dim),
        )

        if num_classes is None:
            self.label_embedding = None
        else:
            self.label_embedding = nn.Embedding(num_classes, time_embedding_dim)

        make_res_block = lambda in_c, out_c, **kwargs: ResBlock(
            in_c,
            out_c,
            signal_dim=signal_dim,
            norm_eps=1.0e-5,
            time_embedding_channels=time_embedding_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            **kwargs,
        )

        def make_attn_block(in_c: int) -> nn.Module:
            err_msg = "either `num_heads` or `num_head_channels` should be provided"
            if num_head_channels is not None:
                n_heads = in_c // num_head_channels
            else:
                if num_heads is None:
                    raise ValueError(err_msg)
                n_heads = num_heads
            head_c = in_c // n_heads if num_head_channels is None else num_head_channels
            if not use_spatial_transformer:
                # TODO: support attn_split_chunk
                return MultiHeadSpatialAttention(
                    in_c,
                    num_heads=n_heads,
                    num_head_channels=head_c,
                    use_checkpoint=use_checkpoint,
                )
            return SpatialTransformer(
                in_c,
                n_heads,
                head_c,
                num_layers=num_transformer_layers,
                context_dim=context_dim,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                attn_split_chunk=attn_split_chunk,
                tome_info=tome_info,
            )

        def make_downsample(in_c: int, out_c: int) -> TimestepAttnSequential:
            if not resample_with_resblock:
                return TimestepAttnSequential(
                    ResDownsample(
                        in_c,
                        resample_with_conv,
                        signal_dim=signal_dim,
                        out_channels=out_c,
                    )
                )
            res_block = make_res_block(in_c, out_c, integrate_downsample=True)
            return TimestepAttnSequential(res_block)

        # input
        input_blocks = [
            TimestepAttnSequential(
                conv_nd(signal_dim, in_channels, start_channels, 3, padding=1)
            ),
        ]
        zero_convs = [self.make_zero_conv(start_channels)]

        self.input_hint_block = TimestepAttnSequential(
            conv_nd(signal_dim, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(signal_dim, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(signal_dim, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(signal_dim, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(signal_dim, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(signal_dim, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(signal_dim, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(signal_dim, 256, start_channels, 3, padding=1)),
        )

        feature_size = start_channels
        input_block_channels = [start_channels]
        in_nc = start_channels
        downsample_rate = 1
        for i, multiplier in enumerate(channel_multipliers):
            for _ in range(num_res_blocks):
                out_nc = multiplier * start_channels
                blocks = [make_res_block(in_nc, out_nc)]
                in_nc = out_nc
                if downsample_rate in attention_downsample_rates:
                    blocks.append(make_attn_block(in_nc))
                input_blocks.append(TimestepAttnSequential(*blocks))
                zero_convs.append(self.make_zero_conv(in_nc))
                feature_size += in_nc
                input_block_channels.append(in_nc)
            if i != len(channel_multipliers) - 1:
                out_nc = in_nc
                input_blocks.append(make_downsample(in_nc, out_nc))
                zero_convs.append(self.make_zero_conv(in_nc))
                downsample_rate *= 2
                feature_size += in_nc
                input_block_channels.append(in_nc)

        self.input_blocks = nn.ModuleList(input_blocks)
        self.zero_convs = nn.ModuleList(zero_convs)

        # middle_block
        self.middle_block = TimestepAttnSequential(
            make_res_block(in_nc, in_nc),
            make_attn_block(in_nc),
            make_res_block(in_nc, in_nc),
        )
        self.middle_block_out = self.make_zero_conv(in_nc)
        feature_size += in_nc

    def make_zero_conv(self, channels: int) -> TimestepAttnSequential:
        return TimestepAttnSequential(
            zero_module(conv_nd(self.signal_dim, channels, channels, 1, padding=0))
        )

    def set_tome_info(self, tome_info: Optional[Dict[str, Any]]) -> None:
        for m in self.modules():
            if isinstance(m, SpatialTransformer):
                m.set_tome_info(tome_info)

    def forward(
        self,
        net: Tensor,
        hint: Tensor,
        *,
        timesteps: Tensor,
        context: Optional[Tensor] = None,
    ) -> List[Tensor]:
        # tomesd
        for m in self.modules():
            if isinstance(m, SpatialTransformer):
                for block in m.blocks:
                    if block.tome_info is not None:
                        block.tome_info["size"] = net.shape[-2:]

        # timenet
        time_net = timestep_embedding(
            timesteps,
            self.start_channels,
            dtype=net.dtype,
            repeat_only=False,
        )
        time_net = self.time_embed(time_net)
        guided_hint = self.input_hint_block(hint, time_net, context)

        # main
        nets = []
        for block, zero_conv in zip(self.input_blocks, self.zero_convs):
            net = block(net, time_net, context)
            if guided_hint is not None:
                net += guided_hint
                guided_hint = None
            nets.append(zero_conv(net, time_net, context))

        net = self.middle_block(net, time_net, context)
        net = self.middle_block_out(net, time_net, context)
        nets.append(net)

        return nets


__all__ = [
    "UNetDiffuser",
    "ControlNet",
]
