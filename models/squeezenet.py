import torch
import torch.nn as nn

__all__ = ["SqueezeNet", "squeezenet1_0", "squeezenet1_1"]


class FireModule(nn.Module):
    def __init__(self, in_planes, out_planes, sr = 0.125, pct3x3 = 0.5, res = False):
        super().__init__()
        self.s1x1 = nn.Conv2d(in_planes, out_planes * sr, 1, 1)
        self.e1x1 = nn.Conv2d(out_planes * sr, out_planes * (1-pct3x3), 1, 1)
        self.e3x3 = nn.Conv2d(out_planes * sr, out_planes * pct3x3, 1, 1, padding = 1)
        self.act = nn.ReLU()
        self.res = res

        if self.res:
            assert in_planes == out_planes, \
                f"residual connection cannot be implemented as in_planes({in_planes})!=out_planes({out_planes})"

    def forward(self, x):

        output = self.act(self.s1x1(output))
        output = torch.cat([self.e1x1(output), self.e3x3(output)], dim = 1)
        output = self.act(output)
        if self.res:
            # note this is different from ResNet's skip connection which takes place before activation
            # in paper:
            # "we set the input to Fire4 equal to output of Fire2 + output of Fire3"
            # x here is the output from the previous layer
            output = output + x
        return output



class SqueezeNet(nn.Module):
    def __init__(self, num_classes = 1000, use_res = False, conv1_filters = 96, conv1_ks = 7, conv1_pad = 2, 
            pool_pos = [1, 4, 8], base_e = 128, incr_e = 128, pct3x3 = 0.5, freq = 2, sr = 0.125):
        super().__init__()

        layers = []
        prev_planes = 3
        for i in range(1, 10):
            if i == 1:
                out_planes = conv1_filters
                layers.append(nn.Conv2d(prev_planes, out_planes, conv2_ks, 2, padding = conv1_pad))
            else:
                fire_idx = i - 1
                out_planes = base_e + (fire_idx // freq) * incre_e
                if use_res and prev_planes == out_planes:
                    res = True
                layers.append(FireModule(prev_planes, out_planes, sr, pct3x3, res))
            prev_planes = out_planes

            if i in pool_pos:
                layers.append(nn.MaxPool2d((3, 2)))
        
        self.features = nn.Sequential(**layers)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(prev_planes, num_classes, 1, 1),
            nn.AdaptiveAvgPool((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x



configs = {
    "1.0": {"conv1_filters": 96, "conv1_ks": 7, "conv1_pad": 2, "pool_pos": [1, 4, 8]},
    "1.1": {"conv1_filters": 64, "conv1_ks": 3, "conv1_pad": 0, "pool_pos": [1, 3, 5]}
}


def squeezenet1_0(num_classes = 1000, use_res = False):
    cfg = configs["1.0"]
    return SqueezeNet(num_classes, use_res, **cfg)


def squeezenet1_1(num_classes = 1000, use_res = False):
    cfg = configs["1.1"]
    return SqueezeNet(num_classes, use_res, **cfg)