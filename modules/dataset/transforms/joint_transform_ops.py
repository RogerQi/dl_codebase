import random
import numpy as np
import torch
import torch.nn.functional as F
from .transforms_registry import registry

joint_transforms_registry = registry()

######################
# Crop
######################
@joint_transforms_registry.register
def joint_random_crop(transforms_cfg):
    output_H, output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    def crop(img, target):
        assert img.shape[-2:] == target.shape[-2:]
        assert len(img.shape) == 3, "Only C x H x W images are supported"
        assert len(target.shape) == 2, "Only H x W labels are supported"
        H, W = img.shape[-2:]
        if H <= output_H or W <= output_W:
            # Pad and crop from zero
            temp_size = (max(H, output_H), max(W, output_W))
            temp_shape = img.shape[:-2] + temp_size
            img_bg = torch.zeros(temp_shape, dtype = img.dtype, device = img.device)
            target_bg = torch.zeros(temp_size, dtype = target.dtype, device = target.device)
            img_bg[:, 0:H, 0:W] = img
            target_bg[0:H, 0:W] = target
            img = img_bg
            target = target_bg
            i = j = 0
        else:
            i = random.randint(0, H - output_H)
            j = random.randint(0, W - output_W)
        return (img[:, i:i + output_H, j:j + output_W], target[i:i + output_H, j:j + output_W])
    return crop