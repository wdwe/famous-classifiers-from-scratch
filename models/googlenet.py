import torch
import torch.nn as nn
from .utils import Conv2dBn


# TODO
# implement same and valid padding

# https://github.com/tensorflow/tensorflow/issues/18213

# https://github.com/pytorch/pytorch/issues/3867#issuecomment-507010696


# # talk about same/valid padding
# # check initialisation
# class Conv2dDynamicSamePadding(nn.Conv2d):
#     """ 2D Convolutions like TensorFlow, for a dynamic image size """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
#         super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
#         self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

#     def forward(self, x):
#         ih, iw = x.size()[-2:]
#         kh, kw = self.weight.size()[-2:]
#         sh, sw = self.stride
#         oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#         pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
#         pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
#         return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



__all__ = ["GoogLeNet", "googlenet", "GoogLeNetWithLoss"]




class Inception(nn.Module):
    def __init__(
        self,
        in_planes,
        conv1,
        reduce3,
        conv3,
        reduce5,
        conv5,
        proj,
        use_bn = False
    ):
        super().__init__()

        if use_bn:
            conv_block = Conv2dBn
        else:
            conv_block = nn.Conv2d

        self.act = nn.ReLU()
        self.conv1x1 = conv_block(in_planes, conv1, 1, 1)
        self.reduce3 = conv_block(in_planes, reduce3, 1, 1)
        self.conv3x3 = conv_block(reduce3, conv3, 3, 1, padding = 1)
        self.reduce5 = conv_block(in_planes, reduce5, 1, 1)
        # The official torchvision used kernel_size of 3 instead, below is the comment from the
        # official code
        # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
        # Please see https://github.com/pytorch/vision/issues/906 for details.
        self.conv5x5 = conv_block(reduce5, conv5, 5, 1, padding = 2)
        self.pool = nn.MaxPool2d(3, 1, padding = 1, ceil_mode = True)
        self.proj = conv_block(in_planes, proj, 1, 1)

    

    
    def forward(self, x):
        output1x1 = self.conv1x1(x)
        output1x1 = self.act(output1x1)

        output3x3 = self.reduce3(x)
        output3x3 = self.act(output3x3)
        output3x3 = self.conv3x3(output3x3)
        output3x3 = self.act(output3x3)

        output5x5 = self.reduce5(x)
        output5x5 = self.act(output5x5)
        output5x5 = self.conv5x5(output5x5)
        output5x5 = self.act(output5x5)

        proj = self.pool(x)
        proj = self.proj(proj)
        proj = self.act(proj)

        output = torch.cat([output1x1, output3x3, output5x5, proj], dim = 1)
        return output



class SideClassifier(nn.Module):
    def __init__(self, in_planes, num_classes, use_bn = False):
        super().__init__()

        if use_bn:
            conv_block = Conv2dBn
        else:
            conv_block = nn.Conv2d

        self.act = nn.ReLU()
        # self.pool = nn.AvgPool2d(5, 3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv1x1 = conv_block(in_planes, 128, 1, 1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4 * 4 * 128, 1024)
        self.dropout = nn.Dropout(p = 0.7)
        self.linear2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.pool(x)
        x = self.conv1x1(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

# talk about passing blocks instead of bn parameters to simplify the function

class GoogLeNet(nn.Module):
    def __init__(self, num_classes = 1000, use_bn = False):
        # TODO check padding
        super().__init__()

        if use_bn:
            conv_block = Conv2dBn
        else:
            conv_block = nn.Conv2d

        self.stem = nn.Sequential(
            conv_block(3, 64, 7, 2, padding = 3), # 112 x 112
            nn.ReLU(),
            nn.MaxPool2d(3, 2, ceil_mode = True), # 56 x 56
            conv_block(64, 64, 1, 1),
            nn.ReLU(),
            conv_block(64, 192, 3, 1, padding = 1), # 56 x 56
            nn.ReLU(),
            nn.MaxPool2d(3, 2, ceil_mode = True) # 28 x 28, since ceil_mode is True
        )

        self.pool = nn.MaxPool2d(3, 2, ceil_mode = True)

        self.inception_3a = Inception(in_planes = 192, conv1 = 64, 
            reduce3 = 96, conv3 = 128, reduce5 = 16, conv5 = 32, proj = 32, use_bn = use_bn)

        self.inception_3b = Inception(in_planes = 256, conv1 = 128, 
            reduce3 = 128, conv3 = 192, reduce5 = 32, conv5 = 96, proj = 64, use_bn = use_bn)

        self.inception_4a = Inception(in_planes = 480, conv1 = 192, 
            reduce3 = 96, conv3 = 208, reduce5 = 16, conv5 = 48, proj = 64, use_bn = use_bn)

        self.side_classifier_1 = SideClassifier(in_planes = 512, num_classes = num_classes)

        self.inception_4b = Inception(in_planes = 512, conv1 = 160, 
            reduce3 = 112, conv3 = 224, reduce5 = 24, conv5 = 64, proj = 64, use_bn = use_bn)

        self.inception_4c = Inception(in_planes = 512, conv1 = 128, 
            reduce3 = 128, conv3 = 256, reduce5 = 24, conv5 = 64, proj = 64, use_bn = use_bn)

        self.inception_4d = Inception(in_planes = 512, conv1 = 112, 
            reduce3 = 144, conv3 = 288, reduce5 = 32, conv5 = 64, proj = 64, use_bn = use_bn)

        self.side_classifier_2 = SideClassifier(in_planes = 528, num_classes = num_classes)

        self.inception_4e = Inception(in_planes = 528, conv1 = 256, 
            reduce3 = 160, conv3 = 320, reduce5 = 32, conv5 = 128, proj = 128, use_bn = use_bn)

        self.inception_5a = Inception(in_planes = 832, conv1 = 256, 
            reduce3 = 160, conv3 = 320, reduce5 = 32, conv5 = 128, proj = 128, use_bn = use_bn)

        self.inception_5b = Inception(in_planes = 832, conv1 = 384, 
            reduce3 = 192, conv3 = 384, reduce5 = 48, conv5 = 128, proj = 128, use_bn = use_bn)

        self.classifier = nn.Sequential(
            # nn.AvgPool2d(7, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p = 0.4),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # x is of shape 3x224x224
        x = self.stem(x)
        # 192x28x28
        x = self.inception_3a(x)
        # 256x28x28
        x = self.inception_3b(x)
        # 480x28x28
        x = self.pool(x)
        # 480x14x14
        output_4a = self.inception_4a(x)
        # 512x14x14
        x = self.inception_4b(output_4a)
        # 512x14x14
        x = self.inception_4c(x)
        # 528x14x14
        output_4d = self.inception_4d(x)
        x = self.inception_4e(output_4d)
        # 832x14x14
        x = self.pool(x)
        # 832x7x7
        x = self.inception_5a(x)
        # 832x7x7
        x = self.inception_5b(x)
        # 1027x7x7
        x = self.classifier(x)

        if self.training:
            aux_1 = self.side_classifier_1(output_4a)
            aux_2 = self.side_classifier_2(output_4d)
            return aux_1, aux_2, x

        return x



def googlenet(num_classes = 1000, use_bn = True):
    return GoogLeNet(num_classes, use_bn)


class GoogLeNetWithLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = GoogLeNet(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, images, labels):
        if self.model.training:
            aux1, aux2, preds = self.model(images)
            loss = 0.3 * self.criterion(aux1, labels) + 0.3 * self.criterion(aux2, labels) + \
                self.criterion(preds, labels)
        else:
            preds = self.model(images)
            loss = self.criterion(preds, labels)
            
        return preds, loss
