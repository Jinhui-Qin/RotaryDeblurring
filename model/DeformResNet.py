import torch
import torch.nn as nn

from . import common
from .ResNet import DFResNet


def build_model(args):
    print('--------------- Using DeformResNet! ---------------')
    return DeformResNet(args)

class conv_end(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(conv_end, self).__init__()

        modules = [
            common.default_conv(in_channels, out_channels, kernel_size),
            nn.PixelShuffle(ratio)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)

class DeformResNet(nn.Module):
    def __init__(self, args):
        super(DeformResNet, self).__init__()

        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        self.n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = args.kernel_size

        self.n_scales = args.n_scales

        self.body_models = nn.ModuleList([
            DFResNet(args, 3, 3, mean_shift=False),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, DFResNet(args, 6, 3, mean_shift=False))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3, 12)]

    def forward(self, input_pyramid):
        scales = range(self.n_scales-1, -1, -1)

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.n_scales

        input_s = input_pyramid[-1]
        for s in scales:
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid


if __name__ == '__main__':

    
    model = DeformResNet()

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
    print("Number of parameters: %f" % (total))