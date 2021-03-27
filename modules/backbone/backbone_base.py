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
        self.feature_size_ = None
        self.num_channels_ = None
    
    # TODO: make a consistent underlying function get_tensor_shape_
    #       for all following functions to avoid repeated tensor inference
    
    def get_feature_size(self, device_ = None):
        '''
        Return the size of the output tensor computed by the backbone net

        Note: may need to be implement in child classes.
        '''
        if self.feature_size_ is not None:
            # cached
            return self.feature_size_
        self.eval()
        with torch.no_grad():
            input_dim = self._cfg.input_dim
            input_dim = (1,) + input_dim # batch of 1 image
            dummy_tensor = torch.rand(input_dim, device = device_)
            output = self.forward(dummy_tensor)
            self.feature_size_ = np.prod(output.shape)
            return self.feature_size_
    
    def get_num_channels(self, device_ = None):
        '''
        Return the number of channels of the output feature tensor from the backbone
        '''
        if self.num_channels_ is not None:
            # cached
            return self.num_channels_
        self.eval()
        with torch.no_grad():
            input_dim = self._cfg.input_dim
            input_dim = (1,) + input_dim # batch of 1 image
            dummy_tensor = torch.rand(input_dim, device = device_)
            output = self.forward(dummy_tensor) # BxCxHxW
            self.num_channels_ = output.shape[1]
            return self.num_channels_

    def forward(self, x):
        raise NotImplementedError