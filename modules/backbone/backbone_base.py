import torch
import torch.nn as nn

class backbone_base(nn.Module):
    def __init__(self):
        super(backbone_base, self).__init__()
    
    def get_feature_size(self):
        '''
        Need to be implement in child classes.
        
        Return the size of the output tensor computed by the backbone net
        '''
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError