import torch
import torch.nn as nn
import torch.nn.functional as F

class cross_entropy(nn.Module):
    '''
    (Optionally) Weighted cross entropy loss
    '''
    def __init__(self, cfg, loss_weight = None):
        super().__init__()
        self.weight = loss_weight
    
    def forward(self, output, label):
        return F.cross_entropy(output, label, weight = self.weight)

class naive_VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, output, original_input):
        assert output.shape == original_input.shape
        return F.kl_div(output, original_input, reduction = 'mean') + F.mse_loss(output, original_input)