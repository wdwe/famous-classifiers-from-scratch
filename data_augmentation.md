|Common Augmentations | Remark|
|---|---|
|Random Crops | Resize the image's shorter side to 256 and randomly crop 224x224 images|
|Random Horizontal Flipping|Flip the crop with a 50% chance|
|Random Scales|Similar to random crops. However, the shorter size of the image is resized to a value randomly chosen from a range instead of being fixed at 256|
|Random Resized Crops|A crop of random size (default: of 0.08 to 1.0) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size. This is popularly used to train the Inception networks.|
|Color Jittering| Randomly change the brightness, contrast and saturation of an image. |
|PAC Color Jittering|It is a fancier way of applying color jettering. Instead of randomly change the image distribution, noises are generated according to the color channels' principle components. The details are in AlexNet's paper.|
|AutoAugment|A set of augmentation policies (rotation, shear, crop etc) derived using reinforcement learning.|