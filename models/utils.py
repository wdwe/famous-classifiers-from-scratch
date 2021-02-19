import torch
import torch.nn as nn


class Conv2dBn(nn.Module):
    """Conv2d layers appended with Batch Normalisation"""
    def __init__(self, in_planes, out_planes, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x



class ConvHeadEvalModel(nn.Module):
    """A thin wrapper to average the output of a conv-headed model
    across all spatial dimensions
    i.e. a conv-headed model's output has shape [batch_size, num_classes, h, w]
    where h and w are feature map spatial dimension
    This wrapper average the output across the last two dimensions to get shape
    [batch_size, num_classes]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, dim = (2, 3))
        return x



# update, it seems that just remove the data.to(device), nn.DataParallel wiil scatter data
# directly from cpu too, thus this class is redundant
class CustomDataParallel(nn.DataParallel):
    """Force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    Taken and modified from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/4272fd0da2ec4d9fc3075c42cf335005a0aa49a3/utils/utils.py#L183
    """

    def __init__(self, module, device_ids):
        super().__init__(module)
        self.device_ids = device_ids

    def scatter(self, inputs, kwargs, device_ids):
        # Overwriting the scatter method of the parent class. When we use the original pytorch class
        # as in (https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html),
        # the inputs are in fact first send to the default gpu, then split and send to the corresponding
        # gpus. This is kind of redundant and leads greater usage of the default gpu 
        
        
        
        devices = ['cuda:' + str(x) for x in self.device_ids]
        splits = inputs[0].shape[0] // len(self.device_ids)

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        # inputs is a list of tensor inputs, in our case is the images and labels
        # tensor inputs are split
        scattered_images = [inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True) \
            for device_idx in range(len(devices))]

        scattered_labels = [inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True) \
            for device_idx in range(len(devices))]

        scattered_inputs = list(zip(scattered_images, scattered_labels))

        # kwargs arguments are replicated
        replicated_kwargs = [kwargs] * len(devices)

        return scattered_inputs, replicated_kwargs