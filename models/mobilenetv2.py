import torch.nn as nn
from .utils import Conv2dBn


__all__ = ["MobileNetV2", "mobilenet_v2"]

class InvertedBottleneck(nn.Module):
    """Inverted Bottleneck as in the paper.
    Note that the self.pw1 should not be followed by activation.
    """
    def __init__(self, in_planes, out_planes, stride, expansion = 6):
        super().__init__()
        self.res = in_planes == out_planes and stride == 1
        mid_planes = in_planes * expansion
        self.pw1 = nn.Conv2d(in_planes, mid_planes, 1, 1)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.dw = nn.Conv2d(mid_planes, mid_planes, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.pw2 = nn.Conv2d(mid_planes, out_planes, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU()

    def forward(self, x):
        if self.res:
            res = x
        x = self.bn1(self.pw1(x))
        x = self.act(self.bn2(self.dw(x)))
        x = self.act(self.bn3(self.pw2(x)))
        if self.res:
            return x + res
        return x



class MobileNetV2(nn.Module):
    def __init__(self, num_classes, archi = None):
        super().__init__()
        self.pre_bottle = Conv2dBn(3, 32, 3, 2, 1)
        if archi is None:
            archi = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ]
        self.bottlenecks = self._build_bottlenecks(32, archi)
        self.post_bottle = Conv2dBn(archi[-1][1], 1280, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1280, num_classes)
        self.drop = nn.Dropout(0.2)
        self.act = nn.ReLU()
        
    def _build_bottlenecks(self, in_planes, archi):
        layers = []
        for t, c, n, s in archi:
            for _ in range(n):
                layers.append(InvertedBottleneck(in_planes, c, s, t))
                in_planes = c
                s = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.pre_bottle(x))
        x = self.bottlenecks(x)
        x = self.act(self.post_bottle(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
        

def mobilenet_v2(num_classes = 1000):
    return MobileNetV2(num_classes)



if __name__ == "__main__":
    import torch
    m = mobilenet_v2()
    im = torch.randn(32, 3, 224, 224)
    results = m(im)
    print(results.shape)