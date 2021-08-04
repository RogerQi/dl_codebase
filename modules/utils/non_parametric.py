import numpy as np
import torch
import torch.nn.functional as F

def masked_average_pooling(mask_b1hw, feature_bchw, normalization):
    '''
    Params
        - mask_b1hw: a binary mask whose element-wise value is either 0 or 1
        - feature_bchw: feature map obtained from the backbone
    
    Return: Mask-average-pooled vector of shape 1 x C
    '''
    if len(mask_b1hw.shape) == 3:
        mask_b1hw = mask_b1hw.view((mask_b1hw.shape[0], 1, mask_b1hw.shape[1], mask_b1hw.shape[2]))

    # Assert remove mask is not in mask provided
    assert -1 not in mask_b1hw

    # Spatial resolution mismatched. Interpolate feature to match mask size
    if mask_b1hw.shape[-2:] != feature_bchw.shape[-2:]:
        feature_bchw = F.interpolate(feature_bchw, size=mask_b1hw.shape[-2:], mode='bilinear')
    
    if normalization:
        feature_norm = torch.norm(feature_bchw, p=2, dim=1).unsqueeze(1).expand_as(feature_bchw)
        feature_bchw = feature_bchw.div(feature_norm + 1e-5) # avoid div by zero

    batch_pooled_vec = torch.sum(feature_bchw * mask_b1hw, dim = (2, 3)) / (mask_b1hw.sum(dim = (2, 3)) + 1e-5) # B x C
    return torch.mean(batch_pooled_vec, dim=0)