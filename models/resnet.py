import torch
import torch.nn as nn
from .utils import Conv2dBn

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]



class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1):
        super().__init__()
        assert not ((in_planes != out_planes) ^ (stride != 1)), \
            f"in_planes ({in_planes}) != out_planes ({out_planes}) iff stride ({stride}) != 1"
        
        self.conv1 = Conv2dBn(in_planes, out_planes, 3, stride, padding = 1, bias = False)
        self.conv2 = Conv2dBn(in_planes, out_planes, 3, 1, padding = 1, bias = False)
        if stride != 1:
            self.res = Conv2dBn(in_planes, out_planes, 1, stride, padding = 0, bias = False)
        else:
            self.res = nn.Identity()
        

        self.act = nn.ReLU()

    def forward(self, x):
        res = self.res(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x) + res)

        return x



class Bottleneck(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride = 1):
        super().__init__()
        assert not ((in_planes != out_planes) ^ (stride != 1)), \
            f"in_planes ({in_planes}) != out_planes ({out_planes}) iff stride ({stride}) != 1"

        self.squeeze = Conv2dBn(in_planes, mid_planes, 1, stride, bias = False)
        self.conv = Conv2dBn(mid_planes, mid_planes, 3, 1, padding = 1, bias = False)
        self.expand = Conv2dBn(mid_planes, out_planes, 1, 1, bias = False)
        if stride != 1:
            self.res = Conv2dBn(in_planes, out_planes, 1, stride, padding = 0, bias = False)
        else:
            self.res = nn.Identity()

        self.act = nn.ReLU()

    def forward(self, x):
        res = self.res(x)
        x = self.act(self.squeeze(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand(x) + res)

        return x




class ResNet(nn.Module):
    def __init__(self, num_classes = 1000, **kwargs):
        super().__init__()

        self.stem = nn.Sequential(
            Conv2dBn(3, 64, 7, 2, padding = 3, bias = False),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding = 1)
        )

        self.conv_modules, out_planes = self._build_modules(**kwargs, in_planes = 64)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Linear(out_planes, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_modules(x)
        x = self.classifier(x)

        return x

    def _build_module(self, basic_block, num_blocks, planes, stride):
        block_list = []
        for i in range(num_blocks):
            if i == 0:
                block_list.append(basic_block(*planes, stride = stride))
                # now the in_plane should be changed
                planes[0] = planes[-1]
            else:
                block_list.append(basic_block(*planes, stride = 1))

        return nn.Sequential(*block_list)


    def _build_modules(self, basic_block, num_filters, num_blocks, in_planes):
        modules = []
        for i, num_block in enumerate(num_blocks):
            stride = 1 if i == 0 else 2
            modules.append(self._build_module(basic_block, num_block, [in_planes] + num_filters, stride))
            in_planes = num_filters[-1]
            num_filters = [2 * num for num in num_filters]
        
        return nn.Sequential(*modules), in_planes
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_in", nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



configs = {
    18: {
        "basic_block": Block,
        "num_filters": [64],
        "num_blocks": [2, 2, 2, 2]
    },

    34: {
        "basic_block": Block,
        "num_filters": [64],
        "num_blocks": [3, 4, 6, 3]
    },

    50: {
        "basic_block": Bottleneck,
        "num_filters": [64, 256],
        "num_blocks": [3, 4, 6, 3]
    },

    101: {
        "basic_block": Bottleneck,
        "num_filters": [64, 256],
        "num_blocks": [3, 4, 23, 3]
    },

    152: {
        "basic_block": Bottleneck,
        "num_filters": [64, 256],
        "num_blocks": [3, 8, 36, 3]
    },


}



def resnet18(num_classes = 1000):
    return ResNet(num_classes, **configs[18])

def resnet34(num_classes = 1000):
    return ResNet(num_classes, **configs[34])

def resnet50(num_classes = 1000):
    return ResNet(num_classes, **configs[50])

def resnet101(num_classes = 1000):
    return ResNet(num_classes, **configs[101])

def resnet152(num_classes = 1000):
    return ResNet(num_classes, **configs[152])




# configs = {
#     18: {
#         "basic_block": Block,
#         "module1": {
#             "filters": [64],
#             "num_blocks": 2
#         }
#         "module2": {
#             "filters": [128],
#             "num_blocks": 2
#         }
#         "module3": {
#             "filters": [256],
#             "num_blocks": 2
#         }
#         "module4": {
#             "filters": [512],
#             "num_blocks": 2
#         }
#     },
#     34: {
#         "basic_block": Block,
#         "module1": {
#             "filters": [64],
#             "num_blocks": 3
#         }
#         "module2": {
#             "filters": [128],
#             "num_blocks": 4
#         }
#         "module3": {
#             "filters": [256],
#             "num_blocks": 6
#         }
#         "module4": {
#             "filters": [512],
#             "num_blocks": 3
#         }
#     },
#     50: {
#         "basic_block": Bottleneck,
#         "module1": {
#             "filters": [64, 256],
#             "num_blocks": 3
#         }
#         "module2": {
#             "filters": [128, 512],
#             "num_blocks": 4
#         }
#         "module3": {
#             "filters": [256, 1024],
#             "num_blocks": 6
#         }
#         "module4": {
#             "filters": [512, 2048],
#             "num_blocks": 3
#         }
#     },
#     101: {
#         "basic_block": Bottleneck,
#         "module1": {
#             "filters": [64, 256],
#             "num_blocks": 3
#         }
#         "module2": {
#             "filters": [128, 512],
#             "num_blocks": 4
#         }
#         "module3": {
#             "filters": [256, 1024],
#             "num_blocks": 23
#         }
#         "module4": {
#             "filters": [512, 2048],
#             "num_blocks": 3
#         }
#     },
#     152: {
#         "basic_block": Bottleneck,
#         "module1": {
#             "filters": [64, 256],
#             "num_blocks": 3
#         }
#         "module2": {
#             "filters": [128, 512],
#             "num_blocks": 8
#         }
#         "module3": {
#             "filters": [256, 1024],
#             "num_blocks": 36
#         }
#         "module4": {
#             "filters": [512, 2048],
#             "num_blocks": 3
#         }
#     }
# }
