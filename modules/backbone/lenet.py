import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_base import backbone_base
from utils import conv_output_shape

class net(backbone_base):
    '''
    Implementation of LeNet from Pytorch official.

    Good for testing if the pipeline is working.
    '''
    def __init__(self, cfg):
        if len(cfg.input_dim) == 3:
            in_channel, h, w = cfg.input_dim
        elif len(cfg.input_dim) == 2:
            in_channel = 1
            h, w = cfg.input_dim
        else:
            raise NotImplementedError("LeNet does not recognize input dim: {}".format(cfg.input_dim))
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1)
        h, w = conv_output_shape(h, w, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        h, w = conv_output_shape(h, w, 3, 1)
        self.feature_size = int(h * w * 16)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x
    
    def get_feature_size(self):
        return self.feature_size