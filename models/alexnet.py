import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000, head = "fc"):
        assert head in ["fc", "conv"], "classification head must be fc or conv"
        super().__init__()
        self.features = nn.Sequential(
            # 1st conv
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 11, stride = 4, padding = 2),
            # setting inplace = True, to modify the input directly,
            # this saves some memory but may not be applicable all the time,
            # as the input to nn.ReLU() will be replaced
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # 2nd conv
            nn.Conv2d(64, 192, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # 3rd conv
            nn.Conv2d(192, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            # 4th conv
            # acordding to the paper "One weird trick for parallelizing convolutional neural networks"
            # (on page 5, footnote 1) the 4th conv should have 384 filters
            # but the torchvision put 256 here
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            # 5th conv
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        if head == "fc":
            self.classifier = nn.Sequential(
                # force the feature map to be 6x6x256
                # check https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897
                # for a good explanation for nn.AdaptiveAvgPool2d()
                nn.AdaptiveAvgPool2d((6, 6)),
                # flatten for FC
                nn.Flatten(),
                # FC 1
                nn.Dropout(p = 0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace = True),
                # FC 2
                nn.Dropout(p = 0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                # FC 3
                nn.Linear(4096, num_classes)
            )

        else: # conv head
            self.classifier = nn.Sequential(
                # FC 1 conv implementation
                nn.Dropout(p = 0.5),
                nn.Conv2d(256, 4096, kernel_size = 6, stride = 1),
                nn.ReLU(inplace = True),
                # FC 2 conv implementation
                nn.Dropout(p = 0.5),
                nn.Conv2d(4096, 4096, kernel_size = 1, stride = 1),
                nn.ReLU(inplace = True),
                # FC 3 conv implementation
                nn.Conv2d(4096, num_classes, kernel_size = 1, stride = 1)
            )

    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def alexnet(pretrained = False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        raise Exception("pretrained model is not implemented yet")

    return model

if __name__ == "__main__":
    import torch
    import torchvision
    from collections import OrderedDict

    torch_alex = torchvision.models.alexnet(num_classes = 1000, pretrained = True)
    my_fc_alex = alexnet(num_classes = 1000, head = "fc")
    my_conv_alex = alexnet(num_classes = 1000, head = "conv")

    # print out the parameters to check dimensions
    # model.state_dict() gives an ordered dictionary
    # of the model's parameters and buffers
    # state_dict().items() give all tuples of (name, param/values)
    torch_state_items = list(torch_alex.state_dict().items())
    fc_state_items = list(my_fc_alex.state_dict().items())
    conv_state_items = list(my_conv_alex.state_dict().items())
    for i, item in enumerate(torch_state_items):
        print(f"Torch name: {item[0]:<25}", f"{item[1].shape}")
        print(f"FC name   : {fc_state_items[i][0]:<25}", f"{fc_state_items[i][1].shape}")
        print(f"Conv name : {conv_state_items[i][0]:<25}", f"{conv_state_items[i][1].shape}")
        print("")

    # Store torch model's pretrained weights into an ordered dictionary with
    # our fc_alex model's corresponding parameter name, so that we can load torchvision's
    # pretrained weights into our model for testing
    fc_state_dict = OrderedDict()
    for i, item in enumerate(torch_state_items):
        our_param_name = fc_state_items[i][0]
        pretrained_weight = item[1]
        fc_state_dict[our_param_name] = pretrained_weight

    # torch.save(fc_state_dict, "../weights/alexnet_fc.pth")
    my_fc_alex.load_state_dict(fc_state_dict)

    # Let's do the same for our conv model.
    # However, we need to take care of the dimension this time
    conv_state_dict = OrderedDict()
    for i, item in enumerate(torch_state_items):
        our_param_name = conv_state_items[i][0]
        pretrained_weight = item[1]
        if our_param_name == "classifier.1.weight":
            pretrained_weight = pretrained_weight.view((4096, 256, 6, 6))
        elif our_param_name in ["classifier.4.weight","classifier.6.weight"]:
            # unsqueeze the last dimension twice
            pretrained_weight = pretrained_weight[:, :, None, None]
        conv_state_dict[our_param_name] = pretrained_weight

    # torch.save(conv_state_dict, "../weights/alexnet_conv.pth")
    my_conv_alex.load_state_dict(conv_state_dict)


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
    eval_loader = DataLoader(eval_dataset, batch_size = 50, num_workers = 8)
    device = torch.device("cpu")
    
    # # torch_alex evaluation
    # evaluate(torch_alex, eval_loader, topk = (1, 5), verbose = True, device = device)
    # # Top 1 accuracy: 0.5652200004085899
    # # Top 5 accuracy: 0.7906599985361099 

    # # my_fc_alex evaulation
    # evaluate(my_fc_alex, eval_loader, topk = (1, 5), verbose = True, device = device) 
    # # Top 1 accuracy: 0.5652200004085899
    # # Top 5 accuracy: 0.7906599985361099


    # my_conv_alex evaluation
    from utils import ConvHeadEvalModel
    my_conv_alex = ConvHeadEvalModel(my_conv_alex)
    # center crop evaluation
    # evaluate(my_conv_alex, eval_loader, topk = (1, 5), verbose = True, device = device)
    # Top 1 accuracy: 0.5652200004085899
    # Top 5 accuracy: 0.7906599985361099


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
    eval_loader = DataLoader(eval_dataset, batch_size = 50, num_workers = 8)
    evaluate(my_conv_alex, eval_loader, topk = (1, 5), verbose = True, device = device)
    # Top 1 accuracy: 0.5791200002357364
    # Top 5 accuracy: 0.8041599987149238