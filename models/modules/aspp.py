import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm2d

from .misc import try_index

__all__ = ["ASPP"]


# based on Pytorch official code
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_act, norm):
        if norm_act == 'iabn_sync':
            modules = [
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                norm(out_channels),
            ]
        elif norm_act == 'bn_sync':
            modules = [
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True)
            ]
        else:
            raise NotImplementedError
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_act, norm):
        if norm_act == 'iabn_sync':
            modules = [
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm(out_channels),
            ]
        elif norm_act == 'bn_sync':
            modules = [
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True)
            ]
        else:
            raise NotImplementedError
        super().__init__(*modules)

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels=2048,
        out_channels=256,
        hidden_channels=256,
        output_stride=16,
        norm_act='bn_sync',
        norm=nn.BatchNorm2d,
    ):
        super().__init__()
        modules = []
        self.norm_act = norm_act
        if norm_act == 'iabn_sync':
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    norm(out_channels),
                )
            )
        elif norm_act == 'bn_sync':
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    norm(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        else:
            raise NotImplementedError

        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise NotImplementedError

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, norm_act, norm))
        modules.append(ASPPPooling(in_channels, out_channels, norm_act, norm))

        self.convs = nn.ModuleList(modules)

        if norm_act == 'iabn_sync':
            self.project = nn.Sequential(
                nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                norm(out_channels),
                # nn.Dropout(0.5),
                nn.Dropout(0.1),
            )
            self.last_conv = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                norm(out_channels),
            )

        elif norm_act == 'bn_sync':
            self.project = nn.Sequential(
                nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Dropout(0.1),
            )
            self.last_conv = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
            )

        if norm_act == 'iabn_sync':
            activation = norm(0).activation
            slope = norm(0).activation_param
            self._init_weight((activation, slope))
        elif norm_act == 'bn_sync':
            self._init_weight()

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.last_conv(self.project(res))

    def _init_weight(self, param=None):
        # iabn_sync
        if self.norm_act == 'iabn_sync':
            from inplace_abn import ABN
        if param is not None:
            activation, slope = param
            gain = nn.init.calculate_gain(activation, slope)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight.data, gain)  # Init used in MiB
                elif isinstance(m, ABN):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        # bn_sync
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
