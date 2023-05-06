import random
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import itertools
import torch
from torch import nn as nn
from torch.nn import functional as F

class fuse_img_clinic(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch=1024, out_class=2, dropout=0.3, mode='cat'):
        super(fuse_img_clinic, self).__init__()

        self.mode = mode

        num_modal = len(in_ch)
        if self.mode == 'add':
            self.modality_weights = nn.Parameter(torch.ones(num_modal))

        stages = []
        for in_c in in_ch:
            stages.append(nn.Sequential(
                nn.Linear(in_c, in_c),
                nn.Dropout(dropout),
                nn.Linear(in_c, out_ch)
                ))
        self.stages = nn.ModuleList(stages)

        self.classification = nn.Sequential(
                            nn.Linear(out_ch if self.mode == 'add' else out_ch*num_modal, out_ch),
                            nn.Dropout(dropout),
                            nn.Linear(out_ch, out_class)
                            )

    def forward(self, modalities):

        if self.mode == 'add':
            features = 0.
        else:
            features = []

        for i, x in enumerate(modalities):
            # print(x.shape, len(x), len(modalities))
            x_ = self.stages[i](x)
            # print(x_.shape)
            if self.mode == 'add':
                features += x_ * self.modality_weights[i]
            else:
                if i==0:
                    features = x_
                else:
                    features = torch.cat([features, x_ ], 1)

        out = self.classification(features)

        return out




def normalisation(in_ch, norm='batchnorm', group=32):
    if norm == 'instancenorm':
        norm = nn.InstanceNorm3d(in_ch)
    elif norm == 'groupnorm':
        norm = nn.GroupNorm(group, in_ch)
    elif norm == 'batchnorm':
        norm = nn.BatchNorm3d(in_ch)
    elif callable(activation):
        norm = norm
    else:
        raise ValueError('normalisation type {} is not supported'.format(norm))
    return norm


def activation(act='relu'):
    if act == 'relu':
        a = nn.ReLU(inplace=True)
    elif act == 'lrelu':
        a = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    else:
        raise ValueError('activation type {} is not supported'.format(act))
    return a


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)
        # return input.view(input.size(0), -1)


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, out_channels=None, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.out_channels = out_channels
        if self.out_channels is None:
            self.out_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, out_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        fc_out_3 = fc_out_2.view(batch_size, self.out_channels, 1, 1, 1)

        output_tensor = torch.mul(input_tensor, fc_out_3.expand_as(input_tensor))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv3d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, out_channels=None, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, out_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        cse = self.sSE(input_tensor)
        sse = self.cSE(cse)  # + self.sSE(input_tensor)
        output_tensor = sse + input_tensor
        return F.relu(output_tensor, True)


class ChannelSpatialSELayer3D2(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, out_channels=None, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D2, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, out_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        sse = self.cSE(input_tensor) + self.sSE(input_tensor)
        output_tensor = sse + input_tensor
        return F.relu(output_tensor, True)
