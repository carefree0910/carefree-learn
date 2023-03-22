import math
import torch
import numbers

import torch.nn as nn
import torch.nn.functional as F


class ConvHead(nn.Module):
    def __init__(
        self,
        out_channels,
        in_channels=32,
        num_layers=1,
        kernel_size=3,
        padding=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(ConvHead, self).__init__()
        convhead = []

        for i in range(num_layers):
            convhead.extend(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    norm_layer(in_channels)
                    if norm_layer is not None
                    else nn.Identity(),
                ]
            )
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))

        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class SepConvHead(nn.Module):
    def __init__(
        self,
        num_outputs,
        in_channels,
        mid_channels,
        num_layers=1,
        kernel_size=3,
        padding=1,
        dropout_ratio=0.0,
        dropout_indx=0,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SepConvHead, self).__init__()

        sepconvhead = []

        for i in range(num_layers):
            sepconvhead.append(
                SeparableConv2d(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    dw_kernel=kernel_size,
                    dw_padding=padding,
                    norm_layer=norm_layer,
                    activation="relu",
                )
            )
            if dropout_ratio > 0 and dropout_indx == i:
                sepconvhead.append(nn.Dropout(dropout_ratio))

        sepconvhead.append(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=num_outputs,
                kernel_size=1,
                padding=0,
            )
        )

        self.layers = nn.Sequential(*sepconvhead)

    def forward(self, *inputs):
        x = inputs[0]

        return self.layers(x)


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU
        elif activation.lower() == "softplus":
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dw_kernel,
        dw_padding,
        dw_stride=1,
        activation=None,
        use_bias=False,
        norm_layer=None,
    ):
        super(SeparableConv2d, self).__init__()
        _activation = select_activation_function(activation)
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel,
                stride=dw_stride,
                padding=dw_padding,
                bias=use_bias,
                groups=in_channels,
            ),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias
            ),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            _activation(),
        )

    def forward(self, x):
        return self.body(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm_layer=nn.BatchNorm2d,
        activation=nn.ELU,
        bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class GaussianSmoothing(nn.Module):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    Apply gaussian smoothing on a tensor (1d, 2d, 3d).
    Filtering is performed seperately for each channel in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, padding=0, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1.0
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.0
            kernel *= torch.exp(-(((grid - mean) / std) ** 2) / 2) / (
                std * (2 * math.pi) ** 0.5
            )
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight.
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = torch.repeat_interleave(kernel, channels, 0)

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input, weight=self.weight, padding=self.padding, groups=self.groups
        )


class MaxPoolDownSize(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth):
        super(MaxPoolDownSize, self).__init__()
        self.depth = depth
        self.reduce_conv = ConvBlock(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0
        )
        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    mid_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
                for conv_i in range(depth)
            ]
        )
        self.pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        outputs = []

        output = self.reduce_conv(x)

        for conv_i, conv in enumerate(self.convs):
            output = output if conv_i == 0 else self.pool2d(output)
            outputs.append(conv(output))

        return outputs
