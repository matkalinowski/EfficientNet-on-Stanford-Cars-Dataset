adThis project is thesis code for my postgraduate studies- Deep neural networks, applications in digital media.

Project implements EfficientNet and it's training on Stanford Cars dataset with possibility of extension to both other models and datasets.

- You can find other networks comparison [here](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford)
- You can find original implementation in tensorflow [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
- Relevant experiments, that I performed can be found on [this neptune dashboard](https://ui.neptune.ai/matkalinowski/sandbox/experiments?viewId=47196815-c2a8-4a2b-bb7c-cf3d146805ac).


### Howto

This project uses virutalenv as a package manager. To install it automatically use setup.sh script. I was using python 3.8.5.

#### If you are using windows:
- Make sure that python.exe and pip.exe are in the path system variable. 
    - In case of environment variables order matters. If you have multiple python instances make sure that version 3.8.5 is on top.

#### Common part:
- Go to the project root,
- execute: setup.sh
    - if this does not work you can execute commands line by line

### About EfficientNet

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. EfficientNets are based on AutoML and Compound Scaling. In particular, google first used [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, used the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, our EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.


Details about the supported models are below: 

|    *Name*         |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |      ✓      |
| `efficientnet-b1` |   7.8M   |    78.8    |      ✓      |
| `efficientnet-b2` |   9.2M   |    79.8    |      ✓      |
| `efficientnet-b3` |    12M   |    81.1    |      ✓      |
| `efficientnet-b4` |    19M   |    82.6    |      ✓      |
| `efficientnet-b5` |    30M   |    83.3    |      ✓      |
| `efficientnet-b6` |    43M   |    84.0    |      ✓      |
| `efficientnet-b7` |    66M   |    84.4    |      ✓      |

### Extras
- [lukemelas implementation](https://github.com/lukemelas/EfficientNet-PyTorch), this repository serves also as a source of pretrained network parameters.
- My colleague implementation of other architectures and his results on the same dataset can be found [here](https://github.com/pchaberski/cars). Please note great approach to network configuration. 