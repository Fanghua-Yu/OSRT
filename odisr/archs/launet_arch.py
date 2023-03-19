import torch
import torch.nn as nn
import math


def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DenseBlock(nn.Module):
    def __init__(self, depth=8, rate=8, input_dim=64, out_dims=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        filters = out_dims - rate * depth
        self.dense_module = [
            nn.Sequential(
                nn.Conv2d(input_dim, filters+rate, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        ]

        for i in range(1, depth):
            self.dense_module.append(
                 nn.Sequential(
                    nn.Conv2d(filters+i*rate, rate, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                 )
            )
        self.dense_module = nn.ModuleList(self.dense_module)

    def forward(self, x):
        features = [x]
        x = self.dense_module[0](features[-1])
        features.append(x)
        for idx in range(1, self.depth):
            x = self.dense_module[idx](features[-1])
            features.append(x)
            features[-1] = torch.cat(features[-2:], 1)
        return features[-1]


class CADensenet(nn.Module):
    def __init__(self, conv, n_feat, n_CADenseBlocks=5):
        super(CADensenet, self).__init__()
        self.n_blocks = n_CADenseBlocks

        denseblock = [
            DenseBlock(input_dim=n_feat, out_dims=64) for _ in range(n_CADenseBlocks)]
        calayer = []
        # The rest upsample blocks
        for _ in range(n_CADenseBlocks):
            calayer.append(CALayer(n_feat, reduction=16))

        self.CADenseblock = nn.ModuleList()
        for idx in range(n_CADenseBlocks):
            self.CADenseblock.append(nn.Sequential(denseblock[idx], calayer[idx]))
        self.CADenseblock.append(nn.Conv2d((n_CADenseBlocks+1)*n_feat, n_feat, kernel_size=1))

    def forward(self, x):
        feat = [x]
        for idx in range(self.n_blocks):
            x = self.CADenseblock[idx](feat[-1])
            feat.append(x)
        x = torch.cat(feat[:], 1)
        x = self.CADenseblock[-1](x)
        return x


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def make_model(opt):
    return LAUNet(opt)


class Evaluator(nn.Module):
    def __init__(self, n_feats):
        super(Evaluator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, n_feats, kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2))
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer = nn.Conv2d(in_channels=n_feats, out_channels=2, kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        # saved actions and rewards
        self.saved_action = None
        self.rewards = []
        self.eva_threshold = 0.5

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        if self.training:
            m = torch.distributions.Categorical(softmax)
            action = m.sample()
            self.saved_action = action
        else:
            action = softmax[1]
            action = torch.where(action > self.eva_threshold, 1, 0)
            self.saved_action = action
            m = None
        return action, m


class LAUNet(nn.Module):
    def __init__(self, scale=8, conv=default_conv):
        super(LAUNet, self).__init__()
        self.scale = scale
        self.level = int(math.log(self.scale, 2))
        self.saved_actions = []
        self.softmaxs = []
        n_blocks = 8
        n_feats = 64
        kernel_size = 3
        n_height = 1024
        rgb_range = 255
        n_colors = 3
        n_evaluator = 12
        self.n_evaluator = n_evaluator

        # main SR network
        self.upsample = [nn.Upsample(scale_factor=2 ** (i + 1), mode='bicubic', align_corners=False) for i in
                         range(self.level)]
        self.upsample = nn.ModuleList(self.upsample)

        rgb_mean = (0.4737, 0.4397, 0.4043)
        rgb_std = (1.0, 1.0, 1.0)

        # data preprocessing
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        # head conv
        self.head = conv(n_colors, n_feats)
        # CA Dense net
        self.body = [CADensenet(conv, n_feats, n_CADenseBlocks=(self.level - i) * n_blocks) for i in
                     range(self.level)]
        self.body = nn.ModuleList(self.body)
        # upsample blocks
        self.up_blocks = [Upsampler(default_conv, 2, n_feats, act=False) for _ in
                          range(2 * self.level - 1)]
        self.up_blocks += [Upsampler(default_conv, 2 ** i, 3, act=False) for i in
                           range(self.level - 1, 0, -1)]
        self.up_blocks = nn.ModuleList(self.up_blocks)
        # tail conv that output sr ODIs
        self.tail = [conv(n_feats, n_colors) for _ in range(self.level + 1)]
        self.tail = nn.ModuleList(self.tail)
        # data postprocessing
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        # evaluator subnet
        self.evaluator = nn.ModuleList()
        for p in range(n_evaluator):
            self.evaluator.append(Evaluator(n_feats))

    def merge(self, imglist, radio):
        if radio[0] == 0 and radio[-1] == 0:
            return imglist[-1]
        else:
            result = [imglist[0]]
            for i in range(1, len(imglist)):
                north, middle, south = torch.split(result[-1],
                                                   [radio[0] * i, result[-1].size(2) - radio[0] * i - radio[-1] * i,
                                                    radio[-1] * i], dim=2)
                result.append(torch.cat((north, imglist[i], south), dim=2))
            return result[-1]

    def forward(self, lr):
        results = []
        masks = []
        gprobs = []

        x = self.sub_mean(lr)
        g1 = self.upsample[0](x)
        g2 = self.upsample[1](x)
        g3 = self.upsample[2](x)
        x = self.head(x)
        # 1st level
        b1 = self.body[0](x)
        f1 = self.up_blocks[2](b1)
        f1 = self.tail[0](f1)
        g1 = self.add_mean(f1 + g1)

        eva_g1 = g1.detach()
        patchlist = torch.chunk(eva_g1, self.n_evaluator, dim=2)
        for i in range(len(patchlist)):
            action, gprob = self.evaluator[i](patchlist[i])
            threshold = action.size(0) if self.training else 1
            mask = 1 if int(action.sum()) == threshold else 0
            self.saved_actions.append(action)
            self.softmaxs.append(gprob)
            masks.append(mask)
            gprobs.append(gprob)
        crop_n, remain, crop_s = 0, 0, 0
        for i in range(self.n_evaluator // (2 ** self.level)):
            if masks[i] == 1:
                crop_n += b1.size(2) // self.n_evaluator
            else:
                break
        for j in range(self.n_evaluator - 1, self.n_evaluator * ((2 ** self.level - 1) // (2 ** self.level)),
                       -1):
            if masks[j] == 1:
                crop_s += b1.size(2) // self.n_evaluator
            else:
                break
        remain = b1.size(2) - crop_n - crop_s
        if crop_n or crop_s:
            _, b1re, _ = torch.split(b1, [crop_n, remain, crop_s], dim=2)
            _, g2, _ = torch.split(g2, [crop_n * 4, remain * 4, crop_s * 4], dim=2)
        else:
            b1re = b1
        # 2ed level
        b2 = self.up_blocks[0](b1re)
        b2 = self.body[1](b2)
        f2 = self.up_blocks[3](b2)
        f2 = self.tail[1](f2)
        g2 = self.add_mean(f2 + g2)
        # 3rd level
        if crop_n or crop_s:
            _, b2re, _ = torch.split(b2, [crop_n * 2, b2.size(2) - crop_n * 2 - crop_s * 2, crop_s * 2], dim=2)
            _, g3, _ = torch.split(g3, [crop_n * 16, g3.size(2) - crop_n * 16 - crop_s * 16, crop_s * 16], dim=2)
        else:
            b2re = b2
        b3 = self.up_blocks[1](b2re)
        b3 = self.body[2](b3)
        f3 = self.up_blocks[4](b3)
        f3 = self.tail[2](f3)
        g3 = self.add_mean(f3 + g3)

        g1up = self.up_blocks[5](g1)
        g2up = self.up_blocks[6](g2)
        g4 = self.merge([g1up, g2up, g3], [crop_n * 8, remain * 8, crop_s * 8])
        results = [g1up, g2up, g3, g4]

        return results

if __name__ == '__main__':
    model = LAUNet()
    model(torch.zeros([1, 3, 128, 256]))