import torch.nn as nn

# vgg family configurations
# Note 1: While the paper provides the A-E and A-Local Response Normalisation
# 6 vgg structure. We will only implement A (vgg11), B(vgg13), D(vgg16)
# and E (vgg19) as per torchvision implementation.
# These are also the more popular structures.

# Note 2: Compared to AlexNet, vgg's convolutional layers have more regular
# and consistent structures. It popularised the use of 3x3 conv filters.
# For this reason, we can write a config dictionary to easily specify their
# structures. 

# m in the config indicates a 2x2 max pooling layer with a stride of 2
# number indicates the num of 3x3 kernels (channels of the layer)


# TODO: Write a summary for initialisation for the readers
# read the paper for xavier and kaiming init

# the initialisation method in the pytorch seems to be confusing
# for the fan out... also the linear layers are initialised using that proposed in 
# vgg paper. It is a mix and match

config = {
    "vgg11": [64, "m", 128, "m", 256, 256, "m", 512, 512, "m", 512, 512, "m"],
    "vgg13": [64, 64, "m", 128, 128, "m", 256, 256, "m", 512, 512, "m", 512, 512, "m"],
    "vgg16": [64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"],
    "vgg19": [64, 64, "m", 128, 128, "m", 256, 256, 256, 256, "m", 512, 512, 512, 512, "m", 512, 512, 512, 512, "m"]
}

# for "from vgg import *" in __init__.py
__all__ = [
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'
]

class VGG(nn.Module):
    def __init__(self, name, num_classes = 1000, head = "fc", bn = False, init_weights = True):
        # initiate parent class
        super().__init__()
        assert head in ["fc", "conv"], "classification head must be fc or conv"
        self.features = self._get_conv_layers(name, bn = bn)
        if head == "fc":
            self.classifier = self._get_fc_classifier(num_classes)
        else:
            self.classifier = self._get_conv_classifier(num_classes)

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _get_conv_layers(self, name, bn = False):
        cfg = config[name]
        # As the input image has RGB three channels
        num_prev_channels = 3
        layers = []
        for layer in cfg:
            if layer == "m":
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            else:
                layers.append(nn.Conv2d(num_prev_channels, layer, kernel_size = 3, stride = 1, padding = 1))
                # store the current number of channels
                num_prev_channels = layer
                if bn:
                    # batch normalisation is usually added after convolution
                    # before activations, though some researchers may argue for the
                    # case of putting it after activation
                    layers.append(nn.BatchNorm2d(layer))
                # append ReLU activation
                # again, inplace is set to True to save memory
                layers.append(nn.ReLU(inplace = True))

        # pack all layers into a module
        return nn.Sequential(*layers)

    def _get_fc_classifier(self, num_classes):
        return nn.Sequential(
            # in case the input is not 224 and hence the final
            # feature map has spatial dimensions different from (7, 7)
            nn.AdaptiveAvgPool2d((7, 7)),
            # flatten the feature map for FC layers
            nn.Flatten(),
            # FC 1
            nn.Linear(in_features=7 * 7 * 512, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            # FC 2
            # Dropout is usually added after activation and before convolution
            # again, some researchers may argue for other case
            nn.Dropout(),
            nn.Linear(in_features = 4096, out_features = 4096, bias = True),
            nn.ReLU(inplace = True),
            # FC 3
            nn.Dropout(),
            nn.Linear(in_features = 4096, out_features = num_classes)
        )


    def _get_conv_classifier(self, num_classes):
        return nn.Sequential(
            # Conv for FC 1
            nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = 7),
            nn.ReLU(inplace = True),
            # Conv for FC 2
            nn.Dropout(),
            nn.Conv2d(in_channels = 4096, out_channels = 4096, kernel_size = 1),
            nn.ReLU(inplace = True),
            # Conv for FC3
            nn.Dropout(),
            nn.Conv2d(in_channels = 4096, out_channels = num_classes, kernel_size = 1)
        )
    
    def _init_weights(self):
        # use a mix of Kaiming init and Xavier init
        for module in self.modules():
            # self.modules return an iterators of all the modules
            # Most of the nn layers are subclass of the nn.Module, 
            # as is our network. It iterate in a depth-first manner
            if isinstance(module, nn.Conv2d):
                # if the module that we are currently referring to is an
                # instance of nn.Conv2d, we will use kaiming init
                # https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are
                # https://discuss.pytorch.org/t/how-fan-in-and-fan-out-work-in-torch-nn-init/40013
                # https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L11_weight-init_slides.pdf

                nn.init.kaiming_normal_(module.weight.data, mode = 'fan_in', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                # https://discuss.pytorch.org/t/batchnorm-initialization/16184/2
                # older version of pytorch init nn.BatchNorm2d's weight with uniform
                # distribution and bias with 0. That apparently makes training converge
                # more slowly.
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                nn.init.constant_(module.bias.data, 0)

def vgg11(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg11", num_classes = num_classes, head = head, bn = False)

def vgg13(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg13", num_classes = num_classes, head = head, bn = False)

def vgg16(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg16", num_classes = num_classes, head = head, bn = False)

def vgg19(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg19", num_classes = num_classes, head = head, bn = False)

def vgg11_bn(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg11", num_classes = num_classes, head = head, bn = True)

def vgg13_bn(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg13", num_classes = num_classes, head = head, bn = True)

def vgg16_bn(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg16", num_classes = num_classes, head = head, bn = True)

def vgg19_bn(num_classes = 1000, head = "fc", pretrained = False):
    if pretrained == True:
        raise Exception("Pretrained model is not implemented yet")
    return VGG("vgg19", num_classes = num_classes, head = head, bn = True)


if __name__ == "__main__":
    import torch
    import torchvision
    from collections import OrderedDict
    torch_vgg = torchvision.models.vgg16_bn(num_classes = 1000, pretrained = True)
    my_fc_vgg = vgg16_bn(num_classes = 1000, head = "fc")
    my_conv_vgg = vgg16_bn(num_classes = 1000, head = "conv")

    # print out the parameters to check dimensions
    # model.state_dict() gives an ordered dictionary
    # of the model's parameters and buffers
    # state_dict().items() give all tuples of (name, param/values)
    torch_state_items = list(torch_vgg.state_dict().items())
    fc_state_items = list(my_fc_vgg.state_dict().items())
    conv_state_items = list(my_conv_vgg.state_dict().items())
    for i, item in enumerate(torch_state_items):
        print(f"Torch name: {item[0]:<25}", f"{item[1].shape}")
        print(f"FC name   : {fc_state_items[i][0]:<25}", f"{fc_state_items[i][1].shape}")
        print(f"Conv name : {conv_state_items[i][0]:<25}", f"{conv_state_items[i][1].shape}")
        print("")

    # Store torch model's pretrained weights into an ordered dictionary with
    # our fc_vgg model's corresponding parameter name, so that we can load torchvision's
    # pretrained weights into our model for testing
    fc_state_dict = OrderedDict()
    for i, item in enumerate(torch_state_items):
        our_param_name = fc_state_items[i][0]
        pretrained_weight = item[1]
        fc_state_dict[our_param_name] = pretrained_weight


    # torch.save(fc_state_dict, "../weights/vggnet_fc.pth")
    my_fc_vgg.load_state_dict(fc_state_dict)

    # Let's do the same for our conv model.
    # However, we need to take care of the dimension this time
    conv_state_dict = OrderedDict()
    for i, item in enumerate(torch_state_items):
        our_param_name = conv_state_items[i][0]
        pretrained_weight = item[1]
        if our_param_name == "classifier.0.weight":
            pretrained_weight = pretrained_weight.view((4096, 512, 7, 7))
        elif our_param_name in ["classifier.3.weight","classifier.6.weight"]:
            # unsqueeze the last dimension twice
            pretrained_weight = pretrained_weight[:, :, None, None]
        conv_state_dict[our_param_name] = pretrained_weight

    # torch.save(conv_state_dict, "../weights/vggnet_conv.pth")
    my_conv_vgg.load_state_dict(conv_state_dict)

    # Test our models
    import sys
    import os
    sys.path.insert(0, "..")
    from data.dataset import EvalDataset
    from torch.utils.data import DataLoader
    from tools.eval import evaluate

    imagenet_dir = "../../datasets/imagenet/ILSVRC2015/Data/CLS-LOC/"
    val_dir = os.path.join(imagenet_dir, "val")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dataset_config = {
        "input_size" : 224,
        "imdir" : val_dir, 
        "mean" : mean, 
        "std" : std,
        "rescale_sizes" : [256],
        "center_square" : True,
        "crop" : "center",
        "horizontal_flip" : False,
        "fname" : False
    }

    eval_dataset = EvalDataset(**dataset_config)
    eval_loader = DataLoader(eval_dataset, batch_size = 25, num_workers = 8)
    device = torch.device("cuda:0")

    evaluate(torch_vgg, eval_loader, topk = (1, 5), verbose = True, device = device)
    evaluate(my_fc_vgg, eval_loader, topk = (1, 5), verbose = True, device = device)
    # vgg11
    # Top 1 accuracy: 0.6901999824158848
    # Top 5 accuracy: 0.8862799800634384
    # vgg16
    # Top 1 accuracy: 0.7159200000874698
    # Top 5 accuracy: 0.9038199959695339
    # vgg19
    # Top 1 accuracy: 0.7237599809803069
    # Top 5 accuracy: 0.9087599808275699
    # vgg16_bn
    # Top 1 accuracy: 0.7335999807342887
    # Top 5 accuracy: 0.9151599812954664

    # my_conv_vgg evaluation
    from utils import ConvHeadEvalModel
    my_conv_vgg = ConvHeadEvalModel(my_conv_vgg)
    # center crop evaluation
    evaluate(my_conv_vgg, eval_loader, topk = (1, 5), verbose = True, device = device)
    # vgg11
    # Top 1 accuracy: 0.6901999824158848
    # Top 5 accuracy: 0.8862799800634384
    # vgg16
    # Top 1 accuracy: 0.7159200000874698
    # Top 5 accuracy: 0.9038199959695339
    # vgg19
    # Top 1 accuracy: 0.7237599809803069
    # Top 5 accuracy: 0.9087599808275699
    # vgg16_bn
    # Top 1 accuracy: 0.7335999807342887
    # Top 5 accuracy: 0.9151599812954664


    # The real dense evaluation
    # Note: ideally, the "center_square" should be False so that the network is applied
    # across the uncropped image. However, without "center_square" crop, images are of 
    # different sizes. We therefore cannot stack them into a batch. This is a proof-of
    # -concept implementation.
    dataset_config = {
        "input_size" : 224,
        "imdir" : val_dir, 
        "mean" : mean, 
        "std" : std,
        "rescale_sizes" : [256],
        "center_square" : True,
        "crop" : None,
        "horizontal_flip" : False,
        "fname" : False
    }

    eval_dataset = EvalDataset(**dataset_config)
    eval_loader = DataLoader(eval_dataset, batch_size = 25, num_workers = 8)
    evaluate(my_conv_vgg, eval_loader, topk = (1, 5), verbose = True, device = device)
    # vgg11
    # Top 1 accuracy: 0.6981399813815952
    # Top 5 accuracy: 0.8935599801689387
    # vgg16
    # Top 1 accuracy: 0.7259799807704985
    # Top 5 accuracy: 0.9111999812275171
    # vgg19
    # Top 1 accuracy: 0.7323399809934199
    # Top 5 accuracy: 0.9131399810910225
    # vgg16_bn
    # Top 1 accuracy: 0.7427399804219603
    # Top 5 accuracy: 0.920739981725812