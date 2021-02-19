# Famous Image Classification Networks from Scratch

## Intorduction

In this repository, I build some of the very influential image classification neural networks from scratch, in PyTorch. When I have extra GPU resources to allocate, I will train some of them and try to to reach SOTA performance.<br>
Along the way, I have been blogging about the theories and implementation of these networks. You can check them out at my [medium homepage](wdwe.medium.com) or click the friend links below in the content table.

## Content Table
Networks/Utils | Medium Link | Remarks<sup>1</sup>
---|---|---
:heavy_check_mark: Efficient Evaluation <br>on ImageNEt | [Evaluation](https://medium.com/swlh/scratch-to-sota-build-famous-classification-nets-1-evaluation-dacfe6b29085?source=friends_link&sk=52e45a4c1f5bcdc185792d931d4ea101) |
:heavy_check_mark: Training/Data Parallelism/<br>Distributed Data Parallelism| [Training/<br>(Distributed) Data Parallelism](https://medium.com/swlh/scratch-to-sota-build-famous-classification-nets-3-train-distributed-data-parallelism-1d0527f15df4?source=friends_link&sk=348ff0ec1d5dc21347a7908124231159) |
:heavy_check_mark: AlexNet | [AlexNet/VGG](https://medium.com/swlh/scratch-to-sota-build-famous-classification-nets-2-alexnet-vgg-50a4f55f7f56?source=friends_link&sk=deb432d00bd77b4e3b723b0ee81c6d0a)| 56.67% Accuracy <br> (0.1% :arrow_up: Torchvision pretrained)
:heavy_check_mark: VGG |  [AlexNet/VGG](https://medium.com/swlh/scratch-to-sota-build-famous-classification-nets-2-alexnet-vgg-50a4f55f7f56?source=friends_link&sk=deb432d00bd77b4e3b723b0ee81c6d0a)|
:heavy_check_mark: GoogLeNet | [GoogLeNet](https://medium.com/swlh/scratch-to-sota-build-famous-classification-nets-4-googlenet-47b70899a6ce?source=friends_link&sk=1015b3a1d40cf2d6e967695ca13a9a2a)| 70.07% Accuracy <br> (0.29% :arrow_up: Torchvision Pretrained)
:heavy_check_mark: ResNet | [ResNet](https://wdwe.medium.com/scratch-to-sota-build-famous-classification-nets-5-resnet-dab4f8444a43?source=friends_link&sk=5ab5957c18b7685eb2501dab4e58d684)|
:heavy_check_mark: SqueezeNet 1.0 | coming soon |
:heavy_check_mark: SqueezeNet 1.1 | coming soon |
:heavy_check_mark: MobileNet-v1 | coming soon|
:heavy_check_mark: MobileNet-v2 | coming soon |
:clock5: ShuffleNet | - |
:clock5: Squeeze-and-Excitation | - |
:clock5: EfficientNet | - |

<sup>1</sup> As with Torchvision's models evaluation accuracy, the performance is on ImageNet evaluation set with a single center crop.

## Brief Summary on Common Data Augmentations
### Test Time

| Network    | ImageNet Evaluation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlexNet    | 1. Resize the image's shorter dimension to 256 and take the center crop of 256x256.<br>2. Take 224x224 crops from image center and 4 corners, as well as, their horizonal flips (10 crops).                    <br>3. These 10 crops' softmax outputs are averaged for prediction.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| VGG family | (a) Dense<br>1. Resize the shorter dimension of image to Q<br>2. Run network densely over the image as in Overfeat<br>(mention Andrew Ng video and other references here)<br>3. The prediction feature map is averaged spatially for prediction.<br>(b) Multi-crop<br>1. Resize the shorter dimension of the image to 2 scales (Q1, Q2, Q3).<br>2. For each scale, take regular 5x5 grid crops and their horizontal flips.<br>i.e. The input for network is 224x224. If the resized image is 512x384,then every time we shift the crop location by the rounding of (512-224)/5=57.6horizontally or (384-224)/5=32 vertically, starting from top-left corner.<br>(c) Combined<br>1. Average of the final (averaged) softmax probabilities from (a) and (b)<br>3. The 3x5x5x2=150 crops' softmax probabilities are averaged for prediction. |
| GoogLeNet  | 1. Resize the shorter dimension of the image to 256, 288, 320 and 352<br>2. Take left, center and right squares (for landscape image), or, top,center and bottom squares (for portrait).<br>3. For each square take the 5 224x224 crops as in AlexNet and the wholesquare resized to 224x224, as well as their horizontal flips (12 crops)<br>4. The 4x3x12 = 144 crops' softmax probabilities are averaged for prediction.                                                                                                                                                                                                                                                                                                                                                                                                               |
| ResNet     | (a) 10-crop evaluation as in AlexNet<br>(b) Average the dense (as in VGG (a) above) probabilities formultiple scales, where the images are resized with shorterdimension taking 224, 256, 384, 480, 640                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |



### Training



|   Common Augmentations     |                                          Remarks                                          |
|----------------------------|-------------------------------------------------------------------------------------------|
| Random Crops               | Resize the image's shorter side to 256 and randomly crop it to 224x224.                   |
| Random Horizontal Flipping | Flip the crop with a 50% chance                                                           |
| Random Scales              | Similar to random crops. However, the shorter size of the image is resized to a valuerandomly chosen from a range instead of being fixed at 256.         |
| Random Resized Crops       | A crop of random size (default: of 0.08 to 1.0) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made.  <br> This crop is finally resized to given size.<br> This is popularly used to train the Inception networks.                                   |
| Color Jittering            | Randomly change the brightness, contrast and saturation of an image.                      |
| PAC Color Jittering        | It is a fancier way of applying color jittering.<br>Instead of randomly change the image distribution, noises are generated according to the color channels' principle components.<br>The details are in AlexNet's paper.                                                       |
| AutoAugment                | A set of augmentation policies (rotation, shear, crop etc) derived <br>using reinforcement learning.                                                             |

