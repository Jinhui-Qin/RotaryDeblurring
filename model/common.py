# from model.DCN.modules.deform_conv2d import DeformConv2d
import math
import torchvision.ops
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, deform=False, bias=True, groups=1):
    if deform:
        return DeformableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias, groups=groups)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, groups=groups)

def default_norm(n_feats):
    return nn.BatchNorm2d(n_feats)

def default_act():
    return nn.ReLU(True)

def empty_h(x, n_feats):
    '''
        create an empty hidden state

        input
            x:      B x T x 3 x H x W

        output
            h:      B x C x H/4 x W/4
    '''
    b = x.size(0)
    h, w = x.size()[-2:]
    return x.new_zeros((b, n_feats, h//4, w//4))

class Normalization(nn.Conv2d):
    """Normalize input tensor value with convolutional layer"""
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        super(Normalization, self).__init__(3, 3, kernel_size=1)
        tensor_mean = torch.Tensor(mean)
        tensor_inv_std = torch.Tensor(std).reciprocal()

        self.weight.data = torch.eye(3).mul(tensor_inv_std).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor(-tensor_mean.mul(tensor_inv_std))

        for params in self.parameters():
            params.requires_grad = False

class BasicBlock(nn.Sequential):
    """Convolution layer + Activation layer"""
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        modules = []
        modules.append(
            conv(in_channels, out_channels, kernel_size, bias=bias))
        if norm: modules.append(norm(out_channels))
        if act: modules.append(act())

        super(BasicBlock, self).__init__(*modules)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True, out_feats=False,
        conv=default_conv, norm=False, act=default_act):

        self.out_feats = out_feats
        super(ResBlock, self).__init__()
        out_channels = int(n_feats / 2) if out_feats else n_feats

        modules = []
        modules.append(conv(n_feats, out_channels, kernel_size, bias=bias))
        modules.append(act())
        modules.append(conv(out_channels, n_feats, kernel_size, bias=bias))

        if out_feats:
            head = []
            head.append(conv(n_feats, out_channels, kernel_size, bias=bias))
            self.head = nn.Sequential(*head)

        # for i in range(2):
        #     modules.append(conv(n_feats, out_channels, kernel_size, bias=bias))
        #     if norm: modules.append(norm(out_channels))
        #     if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        if self.out_feats:
            res = self.head(res)

        return res

class ResBlock_mobile(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act, dropout=False):

        super(ResBlock_mobile, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=False, groups=n_feats))
            modules.append(conv(n_feats, n_feats, 1, bias=False))
            if dropout and i == 0: modules.append(nn.Dropout2d(dropout))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feats, 4 * n_feats, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if norm: modules.append(norm(n_feats))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

# Only support 1 / 2
class PixelSort(nn.Module):
    """The inverse operation of PixelShuffle
    Reduces the spatial resolution, increasing the number of channels.
    Currently, scale 0.5 is supported only.
    Later, torch.nn.functional.pixel_sort may be implemented.
    Reference:
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/functional.html#pixel_shuffle
    """
    def __init__(self, upscale_factor=0.5):
        super(PixelSort, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 2, 2, h // 2, w // 2)
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
        x = x.view(b, 4 * c, h // 2, w // 2)

        return x

class Downsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if scale == 0.5:
            modules.append(PixelSort())
            modules.append(conv(4 * n_feats, n_feats, 3, bias))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*modules)

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConv2d, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, 2 * groups * kernel_size * kernel_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                                    kernel_size=kernel_size, padding=padding, dilation=dilation,
                                                    groups=groups, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        return x


class DeformableOffsetConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, offset_groups=3):
        super(DeformableOffsetConv2d, self).__init__()

        print('--------------- Using Multi-Receptive-Field-Offset-Mode! ---------------')

        assert kernel_size == 3

        self.offset_groups_chunk = offset_groups
        if out_channels < offset_groups:
            offset_groups = out_channels
            out_channels = 1
        else:
            out_channels = out_channels // offset_groups

        self.offset_groups = offset_groups
        offset_modules = nn.ModuleList([default_conv(in_channels, 2 * kernel_size * kernel_size, 3)])
        deform_modules = nn.ModuleList([torchvision.ops.DeformConv2d(in_channels=in_channels, out_channels=out_channels,
                                    stride=stride, kernel_size=kernel_size, padding=1, dilation=dilation,
                                    groups=groups, bias=bias)])
        for _ in range(1, self.offset_groups):
            offset_modules += [default_conv(in_channels, 2 * kernel_size * kernel_size, 3)]
            deform_modules += [torchvision.ops.DeformConv2d(in_channels=in_channels, out_channels=out_channels,
                                    stride=stride, kernel_size=kernel_size, padding=1, dilation=dilation,
                                    groups=groups, bias=bias)]
        for o in offset_modules:
            nn.init.constant_(o.weight, 0.)
            nn.init.constant_(o.bias, 0.)
        self.offset_modules = offset_modules
        self.deform_modules = deform_modules

    def forward(self, x):
        x, offset_prior = x
        offset_prior = torch.chunk(offset_prior, self.offset_groups_chunk, 1)
        res = []
        for i in range(self.offset_groups):
            offset = self.offset_modules[i](x)
            res.append(self.deform_modules[i](x, offset_prior[i] + offset))
        res = torch.cat(res, 1)
        return res

