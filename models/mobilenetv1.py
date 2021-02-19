import torch.nn as nn
from .utils import Conv2dBn

__all__ = ["MobileNetV1", "mobilenet_v1"]

class DwSepConv2d(nn.Module):
    """Depth-wise separable convolution.
    Its structure is:
    Depthwise Conv -> Bn -> ReLU -> 1x1 Conv -> Bn -> ReLU.
    """
    def __init__(self, in_planes, out_planes, *args, **kwargs):
        super().__init__()
        kwargs["groups"] = in_planes
        self.dw = nn.Conv2d(in_planes, in_planes, *args, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.pw = nn.Conv2d(in_planes, out_planes, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes = 1000, architecture = None):
        super().__init__()
        self.stem = Conv2dBn(3, 32, 3, 2, padding = 1)
        if architecture is None:
            self.dw_sep_archi = [
                # out_planes, stride, padding
                [64, 1, 1],
                [128, 1, 1],
                [128, 1, 1],
                [256, 1, 1], 
                [256, 1, 1],
                [512, 1, 1],
                *[[512, 1, 1]]*5,
                [1024, 1, 1],
                [1024, 2, 1]
            ]
        else:
            self.dw_sep_archi = architecture
        self.dw_sep_convs = self._get_dw_sep_convs()
        self.fc = nn.Linear(1024, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

    def _get_dw_sep_convs(self):
        dw_sep_convs = []
        in_planes = 32
        for out_planes, stride, padding in self.dw_sep_archi:
            dw_sep_convs.append(DwSepConv2d(in_planes, out_planes, 3, stride, padding))
            in_planes = out_planes
        return nn.Sequential(*dw_sep_convs)

    def forward(self, x):
        x = self.stem(x)
        x = self.act(x)
        x = self.dw_sep_convs(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def mobilenet_v1(num_classes = 1000):
    return MobileNetV1(num_classes)

if __name__ == "__main__":
    import torch
    im = torch.randn((16, 3, 224, 224))
    model = mobilenet_v1(1000)
    a = model(im)
    print(a.shape)