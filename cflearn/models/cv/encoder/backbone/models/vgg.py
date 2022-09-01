import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from torchvision.models import vgg16
from torchvision.models import vgg19

from ..register import register_backbone
from ......misc.toolkit import download_model
from ......modules.blocks import SEBlock


register_backbone(
    "vgg16",
    [64, 128, 256, 512],
    {
        "features.3": "stage0",
        "features.8": "stage1",
        "features.15": "stage2",
        "features.22": "stage3",
    },
)(vgg16)


register_backbone(
    "vgg16_full",
    [64, 128, 256, 512, 512],
    {
        "features.3": "stage0",
        "features.8": "stage1",
        "features.15": "stage2",
        "features.22": "stage3",
        "features.29": "stage4",
    },
)(vgg16)


register_backbone(
    "vgg19",
    [64, 128, 256, 512],
    {
        "features.3": "stage0",
        "features.8": "stage1",
        "features.17": "stage2",
        "features.26": "stage3",
    },
)(vgg19)


register_backbone(
    "vgg19_lite",
    [64, 128, 256, 512],
    {
        "features.1": "stage0",
        "features.6": "stage1",
        "features.11": "stage2",
        "features.20": "stage3",
    },
)(vgg19)


register_backbone(
    "vgg19_large",
    [64, 128, 256, 512, 512, 512],
    {
        "features.3": "stage0",
        "features.8": "stage1",
        "features.17": "stage2",
        "features.22": "stage3_first",
        "features.26": "stage3_second",
        "features.35": "stage4",
    },
)(vgg19)


@register_backbone(
    "vgg_style",
    [64, 128, 256, 512],
    {
        "net.3": "stage0",
        "net.10": "stage1",
        "net.17": "stage2",
        "net.30": "stage3",
    },
)
class VGGStyle(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),
        )
        if pretrained:
            pt_path = download_model("vgg_style")
            self.load_state_dict(torch.load(pt_path))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.net(net)


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        deploy: bool,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_post_se: bool = True,
    ):
        super().__init__()

        kernel_size = self.kernel_size = 3
        padding = self.padding = 1

        self.deploy = deploy
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)

        if not use_post_se:
            self.post_se = nn.Identity()
        else:
            self.post_se = SEBlock(out_channels, out_channels // 4, block_impl="torch")

        if deploy:
            self.conv_fused = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            if out_channels == in_channels and stride == 1:
                self.identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.identity = None
            self.dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.side = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    padding - kernel_size // 2,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, net: Tensor) -> Tensor:
        if self.deploy:
            return self.post_se(self.relu(self.conv_fused(net)))
        if self.identity is None:
            id_out = 0
        else:
            id_out = self.identity(net)
        out = self.dense(net) + self.side(net) + id_out
        return self.post_se(self.relu(out))

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.side)
        kernel_id, bias_id = self._fuse_bn_tensor(self.identity)
        kernel1x1_padded = nn.functional.pad(kernel1x1, [1, 1, 1, 1])
        kernel = kernel3x3 + kernel1x1_padded + kernel_id
        bias = bias3x3 + bias1x1 + bias_id
        return kernel, bias

    def _fuse_bn_tensor(self, branch: Optional[nn.Module]) -> Any:
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv, bn = branch[0], branch[1]
            kernel, running_mean, running_var, gamma, beta, eps = (
                conv.weight,
                bn.running_mean,
                bn.running_var,
                bn.weight,
                bn.bias,
                bn.eps,
            )
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    [self.in_channels, input_dim, 3, 3],
                    dtype=np.float32,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = (
                self.id_tensor,
                branch.running_mean,
                branch.running_var,
                branch.weight,
                branch.bias,
                branch.eps,
            )
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self) -> None:
        if hasattr(self, "conv_fused"):
            return None
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_fused = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.conv_fused.weight.data = kernel
        self.conv_fused.bias.data = bias  # type: ignore
        self.__delattr__("dense")
        self.__delattr__("side")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepVGGStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_blocks: int,
        *,
        deploy: bool = False,
        stride: int = 1,
        dilation: int = 1,
        use_post_se: bool = True,
    ):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        in_nc = in_channels
        out_nc = latent_channels
        for stride in strides:
            blocks.append(
                RepVGGBlock(
                    in_nc,
                    out_nc,
                    deploy=deploy,
                    stride=stride,
                    dilation=dilation,
                    groups=1,
                    use_post_se=use_post_se,
                )
            )
            in_nc = out_nc
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)

    def switch_to_deploy(self) -> None:
        for block in self.net:
            block.switch_to_deploy()


class RepVGG(nn.Module):
    def __init__(
        self,
        num_blocks: List[int],
        width_multiplier: List[float],
        *,
        use_post_se: bool = True,
    ):
        super().__init__()
        deploy = False
        out_channels = min(64, int(64 * width_multiplier[0]))
        self.stage1 = RepVGGBlock(
            3,
            out_channels,
            deploy=deploy,
            stride=2,
            use_post_se=use_post_se,
        )
        self.stage2 = RepVGGStage(
            out_channels,
            int(64 * width_multiplier[0]),
            num_blocks[0],
            deploy=deploy,
            stride=2,
            use_post_se=use_post_se,
        )
        self.stage3 = RepVGGStage(
            int(64 * width_multiplier[0]),
            int(128 * width_multiplier[1]),
            num_blocks[1],
            deploy=deploy,
            stride=2,
            use_post_se=use_post_se,
        )
        self.stage4_first = RepVGGStage(
            int(128 * width_multiplier[1]),
            int(256 * width_multiplier[2]),
            num_blocks[2] // 2,
            deploy=deploy,
            stride=2,
            use_post_se=use_post_se,
        )
        self.stage4_second = RepVGGStage(
            int(256 * width_multiplier[2]),
            int(256 * width_multiplier[2]),
            num_blocks[2] // 2,
            deploy=deploy,
            stride=1,
            use_post_se=use_post_se,
        )
        self.stage5 = RepVGGStage(
            int(256 * width_multiplier[2]),
            int(512 * width_multiplier[3]),
            num_blocks[3],
            deploy=deploy,
            stride=2,
            use_post_se=use_post_se,
        )

    def forward(self, net: Tensor) -> Tensor:
        net = self.stage1(net)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4_first(net)
        net = self.stage4_second(net)
        net = self.stage5(net)
        return net

    def switch_to_deploy(self) -> None:
        self.stage1.switch_to_deploy()
        self.stage2.switch_to_deploy()
        self.stage3.switch_to_deploy()
        self.stage4_first.switch_to_deploy()
        self.stage4_second.switch_to_deploy()
        self.stage5.switch_to_deploy()


@register_backbone(
    "rep_vgg",
    [64, 128, 256, 512, 512, 2048],
    dict(
        stage1="stage1",
        stage2="stage2",
        stage3="stage3",
        stage4_first="stage4_first",
        stage4_second="stage4_second",
        stage5="stage5",
    ),
)
def rep_vgg(pretrained: bool = False) -> RepVGG:
    if pretrained:
        raise ValueError("`RepVGG` does not support `pretrained`")
    return RepVGG([4, 6, 16, 1], [2.0, 2.0, 2.0, 4.0])


@register_backbone(
    "rep_vgg_lite",
    [48, 48, 96, 192, 192, 1280],
    dict(
        stage1="stage1",
        stage2="stage2",
        stage3="stage3",
        stage4_first="stage4_first",
        stage4_second="stage4_second",
        stage5="stage5",
    ),
)
def rep_vgg_lite(pretrained: bool = False) -> RepVGG:
    if pretrained:
        raise ValueError("`RepVGG` does not support `pretrained`")
    return RepVGG([2, 4, 14, 1], [0.75, 0.75, 0.75, 2.5])


@register_backbone(
    "rep_vgg_large",
    [64, 160, 320, 640, 640, 2560],
    dict(
        stage1="stage1",
        stage2="stage2",
        stage3="stage3",
        stage4_first="stage4_first",
        stage4_second="stage4_second",
        stage5="stage5",
    ),
)
def rep_vgg_large(pretrained: bool = False) -> RepVGG:
    if pretrained:
        raise ValueError("`RepVGG` does not support `pretrained`")
    return RepVGG([8, 14, 24, 1], [2.5, 2.5, 2.5, 5.0])


__all__ = [
    "rep_vgg",
    "rep_vgg_lite",
    "rep_vgg_large",
]
