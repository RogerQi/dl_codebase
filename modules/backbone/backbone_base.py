import numpy as np
import torch
import torch.nn as nn

class backbone_base(nn.Module):
    '''
    Base class of backbone
    '''
    def __init__(self, cfg):
        super(backbone_base, self).__init__()
        self._cfg = cfg
        self._feature_size = None
    
    def get_feature_size(self, _device = None):
        '''
        Need to be implement in child classes.
        
        Return the size of the output tensor computed by the backbone net
        '''
        if self._feature_size is not None:
            # cached
            return self._feature_size
        input_dim = self._cfg.input_dim
        input_dim = (1,) + input_dim # batch of 1 image
        dummy_tensor = torch.rand(input_dim, device = _device)
        output = self.forward(dummy_tensor)
        self._feature_size = np.prod(output.shape[1:])
        return self._feature_size

    def forward(self, x):
        raise NotImplementedError