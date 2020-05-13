import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_base import backbone_base

class net(backbone_base):
    '''
    Implementation of LeNet from Pytorch official.

    Good for testing if the pipeline is working.
    '''
    def __init__(self, cfg):
        assert cfg.input_dim == (28, 28)
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x
    
    def get_feature_size(self):
        return 9216