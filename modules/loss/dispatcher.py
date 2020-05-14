import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    loss_name = cfg.LOSS.loss
    assert loss_name != "none"
    if loss_name == "cross_entropy":
        from .loss import cross_entropy
        return cross_entropy(cfg)
    elif loss_name == "naive_vae":
        from .loss import naive_VAE
        return naive_VAE(cfg)
    elif loss_name == "nll_loss":
        from .loss import nll_loss
        return nll_loss(cfg)
    else:
        raise NotImplementedError