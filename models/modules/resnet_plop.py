import torch
import copy
import torch.nn as nn
import torch.nn.functional as functional
from collections import OrderedDict

from .misc import GlobalAvgPool2d, try_index


class ResidualBlock(nn.Module):
    """Configurable residual block

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        groups=1,
        norm_act=nn.BatchNorm2d,
        dropout=None,
        last=False
    ):
        super(ResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                ), ("bn1", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                ), ("bn2", bn2)
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=1, padding=0, bias=False)),
                ("bn1", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation
                    )
                ), ("bn2", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                ("bn3", bn3)
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False
            )
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"

        self._last = last

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x
        x = self.convs(x) + residual

        if self.convs.bn1.activation == "leaky_relu":
            act = functional.leaky_relu(
                x, negative_slope=self.convs.bn1.activation_param, inplace=not self._last
            )
        elif self.convs.bn1.activation == "elu":
            act = functional.elu(x, alpha=self.convs.bn1.activation_param, inplace=not self._last)
        elif self.convs.bn1.activation == "identity":
            act = x

        if self._last:
            return act, x
        return act


class IdentityResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        groups=1,
        norm_act=nn.BatchNorm2d,
        dropout=None
    ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                ), ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                )
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)
                ), ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation
                    )
                ), ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False
            )

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        structure,
        bottleneck,
        norm_act=nn.BatchNorm2d,
        classes=0,
        output_stride=16,
        keep_outputs=False
    ):
        super().__init__()
        self.structure = structure
        self.bottleneck = bottleneck
        self.keep_outputs = keep_outputs

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if output_stride != 8 and output_stride != 16:
            raise ValueError("Output stride must be 8 or 16")

        if output_stride == 16:
            dilation = [1, 1, 1, 2]  # dilated conv for last 3 blocks (9 layers)
        elif output_stride == 8:
            dilation = [1, 1, 2, 4]  # 23+3 blocks (78 layers)
        else:
            raise NotImplementedError

        self.dilation = dilation

        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)), ("bn1", norm_act(64))
        ]
        if try_index(dilation, 0) == 1:
            layers.append(("pool1", nn.MaxPool2d(3, stride=2, padding=1)))
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        ResidualBlock(
                            in_channels,
                            channels,
                            norm_act=norm_act,
                            stride=stride,
                            dilation=dil,
                            last=block_id == num - 1
                        )
                    )
                )

                # Update channels and p_keep
                in_channels = channels[-1]

            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]

        self.out_channels = in_channels

        # Pooling and predictor
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict(
                    [("avg_pool", GlobalAvgPool2d()), ("fc", nn.Linear(in_channels, classes))]
                )
            )

    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = try_index(dilation, mod_id)
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    def forward(self, x):
        outs = []
        attentions = []

        x = self.mod1(x)
        # attentions.append(x)
        outs.append(x)

        x, att = self.mod2(x)
        attentions.append(att)
        outs.append(x)

        x, att = self.mod3(x)
        attentions.append(att)
        outs.append(x)

        x, att = self.mod4(x)
        attentions.append(att)
        outs.append(x)

        x, att = self.mod5(x)
        attentions.append(att)
        outs.append(x)

        if hasattr(self, "classifier"):
            outs.append(self.classifier(outs[-1]))

        if self.keep_outputs:
            return outs, attentions
        else:
            return outs[-1], attentions

    def _load_pretrained_model(self, path: str):
        ckpt = torch.load(path, map_location='cpu')

        for key in copy.deepcopy(list(ckpt['state_dict'].keys())):
            ckpt['state_dict'][key[7:]] = ckpt['state_dict'].pop(key)
        del ckpt['state_dict']['classifier.fc.weight']
        del ckpt['state_dict']['classifier.fc.bias']
        self.load_state_dict(ckpt['state_dict'], strict=False)
        del ckpt  # free memory
