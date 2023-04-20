import torch
import torch.nn as nn
from . import common

def build_model(args):
    return ResNet(args)

class ResNet(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3, n_feats=None, kernel_size=None, n_resblocks=None,
                 mean_shift=True, rotational_blur_field=False):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = args.n_feats if n_feats is None else n_feats
        self.kernel_size = args.kernel_size if kernel_size is None else kernel_size
        self.n_resblocks = args.n_resblocks if n_resblocks is None else n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        modules = []
        modules.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size))
        for _ in range(self.n_resblocks):
            modules.append(common.ResBlock(self.n_feats, self.kernel_size))
        if not rotational_blur_field:
            modules.append(common.default_conv(self.n_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, input):
        if self.mean_shift:
            input = input - self.mean

        output = self.body(input)

        if self.mean_shift:
            output = output + self.mean

        return output


class DFResNet(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3, n_feats=None, kernel_size=None, n_resblocks=None,
                 mean_shift=True):
        super(DFResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = args.n_feats if n_feats is None else n_feats
        self.kernel_size = args.kernel_size if kernel_size is None else kernel_size
        self.n_resblocks = args.n_resblocks if n_resblocks is None else n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        skip_1_1 = []
        skip_1_2 = []
        if args.rotational_deform:
            skip_1_1.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size, deform=True))
            skip_1_2.append(common.default_conv(2 * self.n_feats, self.out_channels, self.kernel_size, deform=True))
        else:
            skip_1_1.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size, deform=False))
            skip_1_2.append(common.default_conv(2 * self.n_feats, self.out_channels, self.kernel_size, deform=False))
        self.skip_1_1 = nn.Sequential(*skip_1_1)
        self.skip_1_2 = nn.Sequential(*skip_1_2)

        skip_2_1 = []
        skip_2_2 = []
        skip_2_3 = []
        for _ in range(int(self.n_resblocks // 3)):
            skip_2_1.append(common.ResBlock(self.n_feats, self.kernel_size))
        for _ in range(int(self.n_resblocks // 3)):
            skip_2_2.append(common.ResBlock(self.n_feats, self.kernel_size))
        skip_2_3.append(common.ResBlock(2 * self.n_feats, self.kernel_size, out_feats=True))
        for _ in range(self.n_resblocks - 2 * int(self.n_resblocks // 3) - 1):
            skip_2_3.append(common.ResBlock(self.n_feats, self.kernel_size))
        self.skip_2_1 = nn.Sequential(*skip_2_1)
        self.skip_2_2 = nn.Sequential(*skip_2_2)
        self.skip_2_3 = nn.Sequential(*skip_2_3)

    def forward(self, input):
        if self.mean_shift:
            input = input - self.mean

        x1_1 = self.skip_1_1(input)
        x2_1 = self.skip_2_1(x1_1)
        x2_2 = self.skip_2_2(x2_1)
        x2 = torch.cat((x2_1, x2_2), 1)
        x2_3 = self.skip_2_3(x2)
        x1 = torch.cat((x1_1, x2_3), 1)
        output = self.skip_1_2(x1)

        if self.mean_shift:
            output = output + self.mean

        return output
