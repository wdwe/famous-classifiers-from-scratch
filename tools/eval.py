import numpy as np
import torch
import torch.nn as nn
import torchvision
from .utils import AverageMeter
from tqdm import tqdm


def accuracy(probabilities, labels, topk = (1, )):
    """return the accuracy for top k"""
    num_images = labels.size()[0]
    max_k = max(topk)
    _, top_cat = probabilities.topk(max_k)
    correct = top_cat == labels.to(top_cat.device)
    accu = [torch.sum(correct[:, :k]).type(torch.float32) / num_images for k in topk]
    return accu


def evaluate(model, dataloader, topk = (1, ), verbose = False, device = None):

    model.eval()
    accu_meters = [AverageMeter() for _ in topk]
    softmax = nn.Softmax(-1)

    if device:
        model = model.to(device)

    with torch.no_grad():
        try:
            for i, data in enumerate(tqdm(dataloader)):
                images = data['image']
                labels = data['label']
                if device:
                    images = images.to(device)
                    labels = labels.to(device)
                # remember our dataset already returns a 4D tensor of [ncorp, c, h, w]
                # dataloader will add a new bs dimension
                bs, ncrops, c, h, w = images.size()
                images = images.view((-1, c, h, w))
                labels = labels.view((bs, 1))
                try:
                    result = model(images)
                except:
                    # in training script later, we may pass a model wrapper 
                    # that also outputs loss, we can ignore this for evaluation
                    result, loss = model(images, torch.flatten(labels))
                    try:
                        loss_meter.update(loss.mean().item(), n = bs)
                    except:
                        loss_meter = AverageMeter()
                        loss_meter.update(loss.mean().item(), n = bs)

                prob = softmax(result)
                prob = prob.view((bs, ncrops, -1))
                prob = torch.mean(prob, dim = 1)
                accu = accuracy(prob, labels, topk = (1, 5))
                # update accu_meters
                for j, meter in enumerate(accu_meters):
                    meter.update(accu[j].item(), n = bs)

        # sometimes, we may wanna quit halfway during evaluation, but are still
        # very curious about the performance for the processed data
        except KeyboardInterrupt:
            if verbose:
                print(f"KeyboardInterrupt after {i} batches")
                for i, meter in enumerate(accu_meters):
                    print(f"Top {topk[i]} accuracy: {meter.avg}")
            raise KeyboardInterrupt

    if verbose:
        for i, meter in enumerate(accu_meters):
            print(f"Top {topk[i]} accuracy: {meter.avg}")
    
    try:
        return accu_meters, loss_meter
    except:
        return accu_meters



if __name__ == "__main__":
    import sys
    import os
    sys.path.append("..")
    from data.dataset import EvalDataset
    from torch.utils.data import DataLoader

    model = torchvision.models.alexnet(pretrained=True)

    imagenet_dir = "../../datasets/imagenet/ILSVRC2015/Data/CLS-LOC/"
    val_dir = os.path.join(imagenet_dir, "val")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dataset_config = {
        "imdir" : val_dir, 
        "mean" : mean, 
        "std" : std,
        "rescale_sizes" : [224, 256, 288],
        "center_square" : False,
        "crop" : "gridcrop",
        "horizontal_flip" : True,
        "fname" : False
    }


    eval_dataset = EvalDataset(**dataset_config)
    eval_loader = DataLoader(eval_dataset, batch_size = 50, num_workers = 8)
    
    evaluate(model, eval_loader, topk = (1, 5), verbose = True, device = torch.device("cuda"))

    # alexnet 1 center crop
    # Top 1 accuracy: 0.5651800000572205
    # Top 5 accuracy: 0.7906999998283386
    # 10 crops
    # Top 1 accuracy: 0.5943400001716613
    # Top 5 accuracy: 0.8127000001144409
    # Fake dense evaluation
    # Top 1 accuracy: 0.5750200000572204
    # Top 5 accuracy: 0.801299999923706
    # Gridcrops with [224, 256, 288]
    # Top 1 accuracy: 0.5999799847789109
    # Top 5 accuracy: 0.8181799780726433


    # vgg16 1 center crop
    # Top 1 accuracy: 0.7159200002861023
    # Top 5 accuracy: 0.9038200002670288
