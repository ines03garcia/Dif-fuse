from pl_bolts.models.autoencoders import AE
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from pl_bolts.models.autoencoders import VAE
import torch
from torch import nn
from torch.nn import functional as F
import os
from argparse import ArgumentParser
from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)




class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate"""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def resize_size_conv3x3(in_planes, out_planes, size = 1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if size == 1:
        return conv3x3(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(size=size), conv3x3(in_planes, out_planes))




def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact"""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))

def resize_size_conv1x1(in_planes, out_planes, size = 1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if size == 1:
        return conv1x1(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(size=size), conv1x1(in_planes, out_planes))


class EncoderBlock(nn.Module):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EncoderBottleneck(nn.Module):
    """
    ResNet bottleneck, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L75
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """
    ResNet block, but convs replaced with resize convs, and channel increase is in
    second conv, not first
    """

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class DecoderBlock1(nn.Module):
    """
    ResNet block, but convs replaced with resize convs, and channel increase is in
    second conv, not first
    """

    expansion = 1

    def __init__(self, inplanes, planes, size=1, upsample=None):
        super().__init__()
        self.conv1 = resize_size_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_size_conv3x3(inplanes, planes, size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class DecoderBottleneck(nn.Module):
    """
    ResNet bottleneck, but convs replaced with resize convs
    """

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetDecoder(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x
class ResNetEncoder_1024(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNetDecoder_1024(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 1024 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 16

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 512, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 256, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 128, layers[2], scale=2)
        self.layer4 = self._make_layer(block, 64, layers[3], scale=2)

        if self.maxpool1:
            self.layer5 = self._make_layer(block, 64, layers[4], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer5 = self._make_layer(block, 64, layers[4])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 1024 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x

class ResNetEncoder_2048(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 2048, layers[5], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNetDecoder_2048(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 2048 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 32

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 1024, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 512, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 256, layers[2], scale=2)
        self.layer4 = self._make_layer(block, 128, layers[3], scale=2)
        self.layer5 = self._make_layer(block, 64, layers[4], scale=2)

        if self.maxpool1:
            self.layer6 = self._make_layer(block, 64, layers[5], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer6 = self._make_layer(block, 64, layers[5])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 2048 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x

def resnet18_encoder(first_conv, maxpool1):
    return ResNetEncoder(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)

def resnet18_encoder_1024(first_conv, maxpool1):
    return ResNetEncoder_1024(EncoderBlock, [2, 2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_1024(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder_1024(DecoderBlock, [2, 2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)

def resnet18_encoder_2048(first_conv, maxpool1):
    return ResNetEncoder_2048(EncoderBlock, [2, 2, 2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_2048(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder_2048(DecoderBlock, [2, 2, 2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)

class AE(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_1024(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 1024,
        latent_dim: int = 512,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_1024, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_1024(first_conv, maxpool1)
        self.decoder = resnet18_decoder_1024(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser



class VAE_1024(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 1024,
        kl_coeff: float = 0.1,
        latent_dim: int = 512,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_1024, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_1024(first_conv, maxpool1)
        self.decoder = resnet18_decoder_1024(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_2048(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 2048,
        latent_dim: int = 1024,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_2048, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_2048(first_conv, maxpool1)
        self.decoder = resnet18_decoder_2048(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=1024)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class VAE_2048(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 2048,
        kl_coeff: float = 0.1,
        latent_dim: int = 1024,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_2048, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_2048(first_conv, maxpool1)
        self.decoder = resnet18_decoder_2048(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=1024)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_1024_512(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_1024_512, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_1024(first_conv, maxpool1)
        self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


class VAE_1024_512(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_1024_512, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_1024(first_conv, maxpool1)
        self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_2048_512(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 2048,
        latent_dim: int = 1024,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_2048_512, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_2048(first_conv, maxpool1)
        self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class VAE_2048_512(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 2048,
        kl_coeff: float = 0.1,
        latent_dim: int = 1024,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_2048_512, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_2048(first_conv, maxpool1)
        self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_512_1024(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_512_1024, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.decoder = resnet18_decoder_1024(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_512_2048(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_512_2048, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.decoder = resnet18_decoder_2048(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_1024_2048(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 1024,
        latent_dim: int = 512,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_1024_2048, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_1024(first_conv, maxpool1)
        self.decoder = resnet18_decoder_2048(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class AE_2048_1024(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 2048,
        latent_dim: int = 1024,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_2048_1024, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_2048(first_conv, maxpool1)
        self.decoder = resnet18_decoder_1024(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=1024,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=2048)

        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class VAE_512_1024(pl.LightningModule):
        """
        Standard VAE with Gaussian Prior and approx posterior.
        Model is available pretrained on different datasets:
        Example::
            # not pretrained
            vae = VAE()
            # pretrained on cifar10
            vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
            # pretrained on stl10
            vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
        """

        pretrained_urls = {
            'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
            'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
        }

        def __init__(
                self,
                input_height: int,
                enc_type: str = 'resnet18',
                first_conv: bool = False,
                maxpool1: bool = False,
                enc_out_dim: int = 512,
                kl_coeff: float = 0.1,
                latent_dim: int = 256,
                lr: float = 1e-4,
                **kwargs
        ):
            """
            Args:
                input_height: height of the images
                enc_type: option between resnet18 or resnet50
                first_conv: use standard kernel_size 7, stride 2 at start or
                    replace it with kernel_size 3, stride 1 conv
                maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
                enc_out_dim: set according to the out_channel count of
                    encoder used (512 for resnet18, 2048 for resnet50)
                kl_coeff: coefficient for kl term of the loss
                latent_dim: dim of latent space
                lr: learning rate for Adam
            """

            super(VAE_512_1024, self).__init__()

            self.save_hyperparameters()

            self.lr = lr
            self.kl_coeff = kl_coeff
            self.enc_out_dim = enc_out_dim
            self.latent_dim = latent_dim
            self.input_height = input_height

            valid_encoders = {
                'resnet18': {
                    'enc': resnet18_encoder,
                    'dec': resnet18_decoder,
                },
                'resnet50': {
                    'enc': resnet50_encoder,
                    'dec': resnet50_decoder,
                },
            }

            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder_1024(self.latent_dim, self.input_height, first_conv, maxpool1)

            self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
            self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        @staticmethod
        def pretrained_weights_available():
            return list(VAE.pretrained_urls.keys())

        def from_pretrained(self, checkpoint_name):
            if checkpoint_name not in VAE.pretrained_urls:
                raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

            return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

        def forward(self, x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return self.decoder(z)

        def _run_step(self, x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return z, self.decoder(z), p, q

        def sample(self, mu, log_var):
            std = torch.exp(log_var / 2)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            return p, q, z

        def step(self, batch, batch_idx):
            x, y = batch
            z, x_hat, p, q = self._run_step(x)

            recon_loss = F.mse_loss(x_hat, x, reduction='mean')

            log_qz = q.log_prob(z)
            log_pz = p.log_prob(z)

            kl = log_qz - log_pz
            kl = kl.mean()
            kl *= self.kl_coeff

            loss = kl + recon_loss

            logs = {
                "recon_loss": recon_loss,
                "kl": kl,
                "loss": loss,
            }
            return loss, logs

        def training_step(self, batch, batch_idx):
            loss, logs = self.step(batch, batch_idx)
            self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, logs = self.step(batch, batch_idx)
            self.log_dict({f"val_{k}": v for k, v in logs.items()})
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)

            parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
            parser.add_argument("--first_conv", action='store_true')
            parser.add_argument("--maxpool1", action='store_true')
            parser.add_argument("--lr", type=float, default=1e-4)

            parser.add_argument(
                "--enc_out_dim",
                type=int,
                default=512,
                help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
            )
            parser.add_argument("--kl_coeff", type=float, default=0.1)
            parser.add_argument("--latent_dim", type=int, default=512)

            parser.add_argument("--batch_size", type=int, default=256)
            parser.add_argument("--num_workers", type=int, default=8)
            parser.add_argument("--data_dir", type=str, default=".")

            return parser

class VAE_512_2048(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_512_2048, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.decoder = resnet18_decoder_2048(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class VAE_1024_2048(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 1024,
        kl_coeff: float = 0.1,
        latent_dim: int = 512,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_1024_2048, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }


        self.encoder = resnet18_encoder_1024(first_conv, maxpool1)
        self.decoder = resnet18_decoder_2048(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=512)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class VAE_2048_1024(pl.LightningModule):
        """
        Standard VAE with Gaussian Prior and approx posterior.
        Model is available pretrained on different datasets:
        Example::
            # not pretrained
            vae = VAE()
            # pretrained on cifar10
            vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
            # pretrained on stl10
            vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
        """

        pretrained_urls = {
            'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
            'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
        }

        def __init__(
                self,
                input_height: int,
                enc_type: str = 'resnet18',
                first_conv: bool = False,
                maxpool1: bool = False,
                enc_out_dim: int = 2048,
                kl_coeff: float = 0.1,
                latent_dim: int = 1024,
                lr: float = 1e-4,
                **kwargs
        ):
            """
            Args:
                input_height: height of the images
                enc_type: option between resnet18 or resnet50
                first_conv: use standard kernel_size 7, stride 2 at start or
                    replace it with kernel_size 3, stride 1 conv
                maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
                enc_out_dim: set according to the out_channel count of
                    encoder used (512 for resnet18, 2048 for resnet50)
                kl_coeff: coefficient for kl term of the loss
                latent_dim: dim of latent space
                lr: learning rate for Adam
            """

            super(VAE_2048_1024, self).__init__()

            self.save_hyperparameters()

            self.lr = lr
            self.kl_coeff = kl_coeff
            self.enc_out_dim = enc_out_dim
            self.latent_dim = latent_dim
            self.input_height = input_height

            valid_encoders = {
                'resnet18': {
                    'enc': resnet18_encoder,
                    'dec': resnet18_decoder,
                },
                'resnet50': {
                    'enc': resnet50_encoder,
                    'dec': resnet50_decoder,
                },
            }

            self.encoder = resnet18_encoder_2048(first_conv, maxpool1)
            self.decoder = resnet18_decoder_1024(self.latent_dim, self.input_height, first_conv, maxpool1)

            self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
            self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        @staticmethod
        def pretrained_weights_available():
            return list(VAE.pretrained_urls.keys())

        def from_pretrained(self, checkpoint_name):
            if checkpoint_name not in VAE.pretrained_urls:
                raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

            return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

        def forward(self, x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return self.decoder(z)

        def _run_step(self, x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return z, self.decoder(z), p, q

        def sample(self, mu, log_var):
            std = torch.exp(log_var / 2)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            return p, q, z

        def step(self, batch, batch_idx):
            x, y = batch
            z, x_hat, p, q = self._run_step(x)

            recon_loss = F.mse_loss(x_hat, x, reduction='mean')

            log_qz = q.log_prob(z)
            log_pz = p.log_prob(z)

            kl = log_qz - log_pz
            kl = kl.mean()
            kl *= self.kl_coeff

            loss = kl + recon_loss

            logs = {
                "recon_loss": recon_loss,
                "kl": kl,
                "loss": loss,
            }
            return loss, logs

        def training_step(self, batch, batch_idx):
            loss, logs = self.step(batch, batch_idx)
            self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, logs = self.step(batch, batch_idx)
            self.log_dict({f"val_{k}": v for k, v in logs.items()})
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)

            parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
            parser.add_argument("--first_conv", action='store_true')
            parser.add_argument("--maxpool1", action='store_true')
            parser.add_argument("--lr", type=float, default=1e-4)

            parser.add_argument(
                "--enc_out_dim",
                type=int,
                default=2048,
                help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
            )
            parser.add_argument("--kl_coeff", type=float, default=0.1)
            parser.add_argument("--latent_dim", type=int, default=1024)

            parser.add_argument("--batch_size", type=int, default=256)
            parser.add_argument("--num_workers", type=int, default=8)
            parser.add_argument("--data_dir", type=str, default=".")

            return parser

class ResNetEncoder_no_bottleneck_6_12_18_24(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 128
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 6, layers[0])
        self.layer2 = self._make_layer(block, 12, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 18, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 24, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class ResNetDecoder_no_bottleneck_6_12_18_24(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 24 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 24, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 18, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 12, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 12, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 12, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        # x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        # x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x

def resnet18_encoder_no_bottleneck_6_12_18_24(first_conv, maxpool1):
    return ResNetEncoder_no_bottleneck_6_12_18_24(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_no_bottleneck_6_12_18_24(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder_no_bottleneck_6_12_18_24(DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)


class AE_no_bottleneck_6_12_18_24(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
            self,
            input_height: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_no_bottleneck_6_12_18_24, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_6_12_18_24(first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_6_12_18_24(self.latent_dim, self.input_height, first_conv, maxpool1)

        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class VAE_no_bottleneck_6_12_18_24(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        batch_size: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        dim_flatten: int = 6144,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE_no_bottleneck_6_12_18_24, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.dim_flatten = dim_flatten
        self.batch_size = batch_size

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_6_12_18_24(first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_6_12_18_24(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.dim_flatten, self.latent_dim)
        self.fc_var = nn.Linear(self.dim_flatten, self.latent_dim)


    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        flat  = x.view(self.batch_size,-1)
        mu = self.fc_mu(flat)
        log_var = self.fc_var(flat)
        p, q, z = self.sample(mu, log_var)
        # return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        #print(x.shape)
        flat  = x.view(self.batch_size,-1)
        #print(flat.shape)
        #print(self.fc_mu)
        mu = self.fc_mu(flat)
        log_var = self.fc_var(flat)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(x), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class ResNetEncoder_no_bottleneck_6_12_16(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 128
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 6, layers[0])
        self.layer2 = self._make_layer(block, 12, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 24, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class ResNetDecoder_no_bottleneck_6_12_16(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 16 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 16, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 12, layers[1], scale=2)
        # self.layer3 = self._make_layer(block, 12, layers[2], scale=2)

        if self.maxpool1:
            self.layer3 = self._make_layer(block, 12, layers[2], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer3 = self._make_layer(block, 12, layers[2])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(12, 4, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        # x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        # x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x



def resnet18_encoder_no_bottleneck_6_12_16(first_conv, maxpool1):
    return ResNetEncoder_no_bottleneck_6_12_16(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_no_bottleneck_6_12_16(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder_no_bottleneck_6_12_16(DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)


class AE_no_bottleneck_6_12_16(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
            self,
            input_height: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_no_bottleneck_6_12_16, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_6_12_16(first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_6_12_16(self.latent_dim, self.input_height, first_conv, maxpool1)

        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        # print('input:', x.shape)
        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)
        # print('output:', x_hat.shape)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


class ResNetEncoder_no_bottleneck_6_12_18_24_large(nn.Module):

    def __init__(self, block, layers, batch_size,first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 128
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.batch_size = batch_size

        if self.first_conv:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 6, layers[0])
        self.layer2 = self._make_layer(block, 12, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 18, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 24, layers[3], stride=2)
        self.layer5 = nn.Linear(6144, 16384)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(self.batch_size, -1)
        x = self.layer5(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class ResNetDecoder_no_bottleneck_6_12_18_24_large(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, batch_size, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 24 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height
        self.batch_size = batch_size

        self.upscale_factor = 8

        self.linear = nn.Linear(16384, 6144)

        self.layer1 = self._make_layer(block, 24, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 18, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 12, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 12, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 12, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        # x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        # x = self.upscale1(x)

        x = self.linear(x)
        x = x.reshape(self.batch_size, 24, 16, 16)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x


def resnet18_encoder_no_bottleneck_6_12_18_24_large(batch_size, first_conv, maxpool1 ):
    return ResNetEncoder_no_bottleneck_6_12_18_24_large(EncoderBlock, [2, 2, 2, 2], batch_size, first_conv, maxpool1)


def resnet18_decoder_no_bottleneck_6_12_18_24_large(latent_dim, input_height, batch_size, first_conv, maxpool1 ):
    return ResNetDecoder_no_bottleneck_6_12_18_24_large(DecoderBlock, [2, 2, 2, 2], 8, 128, batch_size, first_conv, maxpool1)


class AE_no_bottleneck_6_12_18_24_large(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
            self,
            batch_size: int,
            input_height: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_no_bottleneck_6_12_18_24_large, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_6_12_18_24_large(self.batch_size, first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_6_12_18_24_large(self.latent_dim, self.input_height,self.batch_size, first_conv,
                                                                       maxpool1)

        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

class ResNetEncoder_no_bottleneck_8_16_32_64(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 128
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class ResNetDecoder_no_bottleneck_8_16_32_64(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 64 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 64, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 32, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 16, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 16, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 16, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        # x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        # x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x

def resnet18_encoder_no_bottleneck_8_16_32_64(first_conv, maxpool1):
    return ResNetEncoder_no_bottleneck_8_16_32_64(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_no_bottleneck_8_16_32_64(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder_no_bottleneck_8_16_32_64(DecoderBlock, [2, 2, 2, 2], 8, 128, first_conv, maxpool1)

class AE_no_bottleneck_8_16_32_64(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
            self,
            input_height: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_no_bottleneck_8_16_32_64, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_8_16_32_64(first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_8_16_32_64(self.latent_dim, self.input_height, first_conv, maxpool1)

        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


class ResNetEncoder_no_bottleneck_8_16_32_64_anysizeinput(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 128
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class ResNetDecoder_no_bottleneck_8_16_32_64_anysizeinput(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block1, block, layers, latent_dim, input_height,input_width, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 64 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height
        self.input_width = input_width

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer_1(block1, 64, layers[0], size = [int(self.input_height/4),int(self.input_width/4)])
        self.layer2 = self._make_layer(block, 32, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 16, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 16, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 16, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer_1(self, block, planes, blocks, size = 1):
      upsample = None
      upsample = nn.Sequential(
              resize_size_conv3x3(self.inplanes, planes * block.expansion, size),
              nn.BatchNorm2d(planes * block.expansion),
          )

      layers = []
      layers.append(block(self.inplanes, planes, size, upsample))
      self.inplanes = planes * block.expansion
      for _ in range(1, blocks):
          layers.append(block(self.inplanes, planes))

      return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        # x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        # x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x

def resnet18_encoder_no_bottleneck_8_16_32_64_anysizeinput(first_conv, maxpool1):
    return ResNetEncoder_no_bottleneck_8_16_32_64_anysizeinput(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_no_bottleneck_8_16_32_64_anysizeinput(latent_dim, input_height, input_width, first_conv, maxpool1):
    return ResNetDecoder_no_bottleneck_8_16_32_64_anysizeinput(DecoderBlock1, DecoderBlock, [2, 2, 2, 2], 8, 400,500, first_conv, maxpool1)

class AE_no_bottleneck_8_16_32_64_anysizeinput(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    # pretrained_urls = {
    #     'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    # }

    def __init__(
            self,
            input_height: int,
            input_width: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_no_bottleneck_8_16_32_64_anysizeinput, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_8_16_32_64_anysizeinput(first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_8_16_32_64_anysizeinput(self.latent_dim, self.input_height,self.input_width, first_conv, maxpool1)

        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

def conv3x3_3d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3_3d(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if scale == 1:
        return conv3x3_3d(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(scale_factor=scale), conv3x3_3d(in_planes, out_planes))


def resize_size_conv3x3_3d(in_planes, out_planes, size = 1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if size == 1:
        return conv3x3_3d(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(size=size), conv3x3_3d(in_planes, out_planes))




def resize_conv1x1_3d(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact"""
    if scale == 1:
        return conv1x1_3d(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(scale_factor=scale), conv1x1_3d(in_planes, out_planes))

def resize_size_conv1x1_3d(in_planes, out_planes, size = 1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if size == 1:
        return conv1x1_3d(in_planes, out_planes)
    else:
        return nn.Sequential(Interpolate(size=size), conv1x1_3d(in_planes, out_planes))

class DecoderBlock_3d(nn.Module):
    """
    ResNet block, but convs replaced with resize convs, and channel increase is in
    second conv, not first
    """

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3_3d(inplanes, inplanes)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3_3d(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm3d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class EncoderBlock_3d(nn.Module):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder_no_bottleneck_6_12_16_3d(nn.Module):

    def __init__(self,block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 128
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool3d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 6, layers[0])
        self.layer2 = self._make_layer(block, 12, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 24, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_3d(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('input:', x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print('output:', x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class ResNetDecoder_no_bottleneck_6_12_16_3d(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 16 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 16, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 12, layers[1], scale=2)
        # self.layer3 = self._make_layer(block, 12, layers[2], scale=2)

        if self.maxpool1:
            self.layer3 = self._make_layer(block, 12, layers[2], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer3 = self._make_layer(block, 12, layers[2])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv3d(12, 1, kernel_size=(4,3,3), stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1_3d(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        # x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        # x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x



def resnet18_encoder_no_bottleneck_6_12_16_3d(first_conv, maxpool1):
    return ResNetEncoder_no_bottleneck_6_12_16_3d(EncoderBlock_3d, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder_no_bottleneck_6_12_16_3d(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder_no_bottleneck_6_12_16_3d(DecoderBlock_3d, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)


class AE_no_bottleneck_6_12_16_3d(pl.LightningModule):
    """
    Standard AE
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        ae = AE()
        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    # pretrained_urls = {
    #     'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    # }

    def __init__(
            self,
            input_height: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE_no_bottleneck_6_12_16_3d, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        self.encoder = resnet18_encoder_no_bottleneck_6_12_16_3d(first_conv, maxpool1)
        self.decoder = resnet18_decoder_no_bottleneck_6_12_16_3d(self.latent_dim, self.input_height, first_conv, maxpool1)

        # self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        # z = self.fc(feats)
        x_hat = self.decoder(feats)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser
