import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    network_name = cfg.NETWORK.network
    assert cfg.BACKBONE.network == "none"
    assert cfg.CLASSIFIER.classifier == "identity"
    assert network_name != "none"
    if network_name == "unet":
        from .unet import UNet
        return UNet
    else:
        raise NotImplementedError