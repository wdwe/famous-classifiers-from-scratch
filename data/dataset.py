import torch
import PIL
import torchvision
import torchvision.transforms as tsfm
from torchvision.datasets import ImageFolder as ImageFolder

# evaluation datasets and transforms

class ResizeMultiple:
    """Callable class for image transform
    its __call__ takes a PIL image or a list of PIL images and returns
    a list of resized images.
    """
    def __init__(self, rescale_sizes, interpolation = "bilinear"):
        self.sizes = rescale_sizes
        if interpolation == "bilinear":
            self.interpolation = PIL.Image.BILINEAR
        elif interpolation == "bicubic":
            self.interpolation = PIL.Image.BICUBIC
        else:
            raise Exception("Unknown interpolation method")

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        resized_images = []
        for size in self.sizes:
            for image in images:
                resized_images.append(tsfm.functional.resize(image, size, interpolation = self.interpolation))

        return resized_images

class CenterCropMultiple:
    def __init__(self, size = None):
        self.size = size

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        if self.size:
            crops = [tsfm.CenterCrop(self.size)(image) for image in images]
        else:
            crops = [tsfm.CenterCrop(min(image.size))(image) for image in images]

        return crops

class FiveCropMultiple:
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            crops.extend(tsfm.FiveCrop(self.size)(image))

        return crops

class FiveAndOneCropMultiple:
    """Callable class for image transform
    its __call__ takes a PIL image or a list of PIL images and returns
    a list of 5 crops and the image resized to crop size.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            assert image.width == image.height, "image is not square"
            crops.extend(tsfm.FiveCrop(self.size)(image))
            crops.append(tsfm.functional.resize(image, self.size))

        return crops


class ThreeCropMultiple:
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            if image.width > image.height:
                # return left, center, right crops
                size = image.height
                stride = (image.width - size) / 3
                for step in range(3):
                    left = int(step * stride)
                    crops.append(image.crop((left, 0, left + size, size)))
            else:
                # return top, center, right crops
                size = image.width
                stride = (image.height - size) / 3
                for step in range(3):
                    top = int(step * stride)
                    crops.append(image.crop((0, top, size, top + size)))

        return crops


class GridCropMultiple:
    def __init__(self, size, steps = 5):
        self.size = size
        self.steps = steps

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            h = image.height
            w = image.width
            v_stride = (h - self.size) / self.steps
            h_stride = (w - self.size) / self.steps
            for h_step in range(self.steps):
                for v_step in range(self.steps):
                    left = int(h_stride * h_step)
                    top = int(v_stride * v_step)
                    box = (left, top, left + self.size , top + self.size)
                    crops.append(image.crop(box))

        return crops


class HorizontalFlipMultiple:
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        flipped = []
        for image in images:
            flipped.append(PIL.ImageOps.mirror(image))
        return images + flipped

class ToTensorMultiple:
    """Callable class for image transform
    Convert a list of PIL images to a list of tensors
    """
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        images = [tsfm.ToTensor()(image) for image in images]

        return images


class NormalizeMultiple:
    def __init__(self, 
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_images):
        if not isinstance(tensor_images, list):
            tensor_images = [tensor_images]
        return [tsfm.Normalize(self.mean, self.std)(image) for image in tensor_images]

class StackCrops:
    def __call__(self, crops):
        return torch.stack(crops)

class EvalDataset(ImageFolder):
    def __init__(
        self,
        imdir,
        input_size = 224,
        rescale_sizes = [256],
        center_square = False,
        crop = "fivecrop",
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        horizontal_flip = True,
        interpolation = "bilinear",
        fname = False
    ):
        """Initialise the class

        Args:
            imdir (str): The root directory for all images
            input_size (int): The size of network input.
                If crop argument is None, then this argument is ignored.
            rescale_sizes (list): The elements can be either int or tuple.
                Tuple defines the exact size the image to be resized to before cropping.
                Int defines the size the shorter image dimension to be resized to.
                (default is False)
            center_square (bool): Whether to take the center square of the image.
            crop (str, None): Must be of "fivecrop", "gridcrop", "center", "googlenet" or None.
                If this argument is None, the rescaled images are returned without cropping.
            horizontal_flip (bool): Whether to make copies of the crops' horizontal flip.
            interpolation (str): bilinear or bicubic for resize operation
            mean (list): The RGB pixel mean for normalization.
            std (list): The RGB pixel standard deviation for normalization.
            fname (bool): Whether to return filename in __getitem__.
        Returns:
            Dataset whose __getitem__ returns a 4D tensor of [Num_crops, channels, H, W]
        """

        self.imdir = imdir
        self.input_size =  input_size
        self.rescale_sizes = rescale_sizes
        assert crop in ["fivecrop", "googlenet", "center", "gridcrop", None], "crop can only be one of ['fivecrop', 'googlenet', 'center', 'gridcrop', None]"
        self.center_square = center_square
        self.crop = crop
        self.horizontal_flip = horizontal_flip
        self.mean = mean
        self.std = std
        self.fname = fname
        self.interpolation = interpolation
        transforms = self._get_transforms()
        super().__init__(root = self.imdir, transform = transforms)


    def _get_transforms(self):
        transforms = []
        
        transforms.append(ResizeMultiple(self.rescale_sizes, interpolation = self.interpolation))

        if self.center_square:
            transforms.append(CenterCropMultiple())

        if self.crop == "center":
            transforms.append(CenterCropMultiple(self.input_size))
        elif self.crop == "fivecrop":
            transforms.append(FiveCropMultiple(self.input_size))
        elif self.crop == "gridcrop":
            transforms.append(GridCropMultiple(self.input_size))
        elif self.crop == "googlenet":
            transforms.append(ThreeCropMultiple())
            transforms.append(FiveAndOneCropMultiple(self.input_size))

        if self.horizontal_flip:
            transforms.append(HorizontalFlipMultiple())

        transforms.append(ToTensorMultiple())
        transforms.append(NormalizeMultiple(self.mean, self.std))
        # convert the list of tensors to a 4-D tensor of dims
        # [Num_crops, channels, H, W]
        transforms.append(StackCrops())
        transforms = tsfm.Compose(transforms)

        return transforms

    def __getitem__(self, idx):
        tensor, label = super().__getitem__(idx)
        data = {"image": tensor, "label": label}
        if self.fname:
            data["fname"] = self.imgs[idx]
        return data


# training dataset and transformations

class TrainDataset(ImageFolder):
    def __init__(
        self,
        imdir,
        input_size = 224,
        color_jitter = (0.4, 0.4, 0.4),
        resize_scale = (0.08, 1.0),
        ratio = (0.75, 1.333333333),
        interpolation = "bilinear",
        horizontal_flip = True,
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        fname = False
    ):
        if color_jitter is not None:
            assert len(color_jitter) in (3, 4), "color_jitter must be None or in [brightness, contrast, saturation(, hue)]"
        self.color_jitter = color_jitter
        self.input_size = input_size
        self.resize_scale = resize_scale
        self.ratio = ratio
        if interpolation == "bilinear":
            self.interpolation = PIL.Image.BILINEAR
        elif interpolation == "bicubic":
            self.interpolation = PIL.Image.BICUBIC
        else:
            raise Exception("interpolation must be either bilinear or bicubic")
        self.horizontal_flip = horizontal_flip
        self.mean = mean
        self.std = std
        self.fname = fname
        transforms = self._get_transforms()
        super().__init__(root = imdir, transform = transforms)

    
    def _get_transforms(self):
        transforms = []
        # add random resiced crop
        transforms.append(
            tsfm.RandomResizedCrop(self.input_size, scale=self.resize_scale, ratio=self.ratio, interpolation=self.interpolation)
        )
        # randomly flip the crop horizontally
        if self.horizontal_flip:
            transforms.append(tsfm.RandomHorizontalFlip(p = 0.5))
        # apply color jittering
        if self.color_jitter:
            color_args = ["brightness", "contrast", "saturation", "hue"]
            color_kwargs = dict(zip(color_args, self.color_jitter))
            transforms.append(
                tsfm.ColorJitter(**color_kwargs)
            )
        # conver the crops from PIL.Image to tensor
        # Note that the axes are transposed from [H, W, C] to [C, H, W]
        transforms.append(tsfm.ToTensor())
        # normalize
        transforms.append(tsfm.Normalize(mean = self.mean, std = self.std))
        
        return tsfm.Compose(transforms)
    
    def __getitem__(self, idx):
        tensor, label = super().__getitem__(idx)
        data = {"image": tensor, "label": label}
        if self.fname:
            data["fname"] = self.imgs[idx]
        return data

if __name__ == "__main__":
    # # eval dataset testing
    import os
    from tensorboardX import SummaryWriter
    # writer = SummaryWriter("../logs/dataset/eval/tensorboard/")
    imagenet_dir = "../../datasets/imagenet/ILSVRC2015/Data/CLS-LOC/"
    # val_dir = os.path.join(imagenet_dir, "val")
    # fname = True
    
    # # imagenet mean and std
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # # create an eval config dictionary for testing
    # eval_configs = {}
    # eval_configs["AlexNet"] = {
    #     "imdir" : val_dir, 
    #     "mean" : mean, 
    #     "std" : std,
    #     "rescale_sizes" : [256],
    #     "center_square" : True,
    #     "crop" : "fivecrop",
    #     "horizontal_flip" : True,
    #     "fname" : fname
    # }

    # eval_configs["VGG16_dense"] = {
    #     "imdir" : val_dir, 
    #     "mean" : mean, 
    #     "std" : std,
    #     "rescale_sizes" : [256],
    #     "center_square" : False,
    #     "crop" : None,
    #     "horizontal_flip" : False,
    #     "fname" : fname
    # }

    # eval_configs["VGG16_Multi_Crop"] = {
    #     "imdir" : val_dir, 
    #     "input_size" : 224,
    #     "mean" : mean, 
    #     "std" : std,
    #     "rescale_sizes" : [224, 256, 288],
    #     "center_square" : False,
    #     "crop" : "gridcrop",
    #     "horizontal_flip" : True,
    #     "fname" : fname
    # }

    # eval_configs["GoogLeNet"] = {
    #     "imdir" : val_dir, 
    #     "input_size" : 224,
    #     "mean" : mean, 
    #     "std" : std,
    #     "rescale_sizes" : [256, 288, 320, 352],
    #     "center_square" : False,
    #     "crop" : "googlenet",
    #     "horizontal_flip" : True,
    #     "fname" : fname
    # }
    
    # # Checking different eval settings
    # for eval_name in eval_configs:
    #     dataset = EvalDataset(
    #         **eval_configs[eval_name]
    #         )
        
    #     print(f"{eval_name} eval shape: ", dataset[678]['image'].shape)
    #     print("Label: ", dataset[678]['label'])
    #     if fname:
    #         print("Filename is: ", dataset[678]['fname'])

    #     images = dataset[678]['image']
    #     images = torchvision.utils.make_grid(images, normalize = True)
    #     writer.add_image(eval_name, images, global_step = 0)

    # # Remember to close the writer to flush all unfinished write operation
    # writer.close()


    # training dataset testing
    import random
    import cv2
    import numpy as np

    train_dir = os.path.join(imagenet_dir, "train")
    train_config = {
        "imdir" : train_dir,
        "input_size" : 224,
        "color_jitter" : (0.4, 0.4, 0.4),
        "resize_scale" : (0.08, 1.0),
        "ratio" : (0.75, 1.333333333),
        "interpolation" : "bilinear",
        "horizontal_flip" : True,
        "mean" : [0.485, 0.456, 0.406],
        "std" : [0.229, 0.224, 0.225],
        "fname" : True
    }


    dataset = TrainDataset(**train_config)
    writer = SummaryWriter("../logs/dataset/train/tensorboard/")
    
    # randomly checking 10 training crops
    for step in range(10):
        idx = random.randint(0, len(dataset) - 1)
        data = dataset[idx]
        image = data["image"]
        label = data["label"]
        filename = data["fname"][0]
        # use utils.make_grid to rescale the image back
        # It takes a 4D tensor or a list of tensors
        image = torchvision.utils.make_grid([image], normalize = True)
        # load the original image
        original_image = np.array(PIL.Image.open(filename))
        # transpose the axes from [H, W, C] to [C, H, W]
        original_image = np.transpose(original_image, (2, 0, 1))
        writer.add_image("training_crop", image, global_step = step)
        writer.add_image("original_image", original_image, global_step = step)
    # remember to close the writer to flush all the write operation
    writer.close()





