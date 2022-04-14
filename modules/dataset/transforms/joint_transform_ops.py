import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as tr_F
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
        H, W = img.shape[-2:]
        if H <= output_H or W <= output_W:
            # Pad and crop from zero
            temp_size = (max(H, output_H), max(W, output_W))
            temp_shape = img.shape[:-2] + temp_size
            img_bg = torch.zeros(temp_shape, dtype = img.dtype, device = img.device)
            img_bg[:, 0:H, 0:W] = img
            img = img_bg
            if len(target.shape) == 2:
                target_bg = torch.zeros(temp_size, dtype = target.dtype, device = target.device)
                target_bg[0:H, 0:W] = target
                target = target_bg
            else:
                assert len(target.shape) == 3, "This block supports C x H x W binary labels"
                C = target.shape[0]
                target_bg = torch.zeros((C,) + temp_size, dtype = target.dtype, device = target.device)
                target_bg[:, 0:H, 0:W] = target
                target = target_bg
            i = j = 0
        else:
            i = random.randint(0, H - output_H)
            j = random.randint(0, W - output_W)
        if len(target.shape) == 2:
            return (img[:, i:i + output_H, j:j + output_W], target[i:i + output_H, j:j + output_W])
        else:
            return (img[:, i:i + output_H, j:j + output_W], target[:, i:i + output_H, j:j + output_W])
    return crop

# @joint_transforms_registry.register
# Though this is helpful in classification, empirically it seems to hurt segmentation performance.
# (possibly due to loss of context)
# Hence, it is excluded from the registry for now to prevent misuse.
def joint_random_resized_crop(transforms_cfg):
    output_H, output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    cropper = torchvision.transforms.RandomResizedCrop((output_H, output_W))
    def crop(img, target):
        assert img.shape[-2:] == target.shape[-2:]
        assert len(img.shape) == 3, "Only C x H x W images are supported"
        if len(target.shape) == 2:
            target = target.view((1,) + target.shape)
        i, j, h, w = cropper.get_params(img, cropper.scale, cropper.ratio)
        img = tr_F.resized_crop(img, i, j, h, w, cropper.size, torchvision.transforms.InterpolationMode.BILINEAR)
        target = tr_F.resized_crop(target, i, j, h, w, cropper.size, torchvision.transforms.InterpolationMode.NEAREST) # 1 x H x W
        target = target.view(target.shape[1:])
        return (img, target)
    return crop

@joint_transforms_registry.register
def joint_naive_resize(transforms_cfg):
    output_H, output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    size = (output_H, output_W)
    def crop(img, target):
        assert img.shape[-2:] == target.shape[-2:]
        img = tr_F.resize(img, size)
        if len(target.shape) == 2:
            # HxW?
            target = target.reshape((1,) + target.shape)
            target = tr_F.resize(target, size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            assert target.shape[0] == 1
            target = target.reshape(target.shape[1:])
        else:
            target = tr_F.resize(target, size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return (img, target)
    return crop

@joint_transforms_registry.register
def joint_random_scale_crop(transforms_cfg):
    output_H, output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    size = (output_H, output_W)
    def crop(img, target):
        assert img.shape[-2:] == target.shape[-2:], "image and label map size mismatched"
        img_H = img.shape[-2]
        img_W = img.shape[-1]
        # Random scale
        scale = np.random.uniform(0.5, 2)
        target_H, target_W = int(img_H * scale), int(img_W * scale)
        img = tr_F.resize(img, (target_H, target_W))
        assert len(target.shape) == 2
        target = target.view((1,) + target.shape)
        target = tr_F.resize(target, (target_H, target_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        # Random crop
        H_padding = int(max(0, np.ceil((output_H - target_H) / 2)))
        W_padding = int(max(0, np.ceil((output_W - target_W) / 2)))
        img = tr_F.pad(img, (W_padding, H_padding), 0, 'constant')
        target = tr_F.pad(target, (W_padding, H_padding), 0, 'constant')
        # Restore target label map to (H, W)
        target = target.view(target.shape[1:])
        start_x = random.randint(0, target.shape[1] - output_W)
        start_y = random.randint(0, target.shape[0] - output_H)
        img = img[:,start_y:start_y+output_H,start_x:start_x+output_W]
        target = target[start_y:start_y+output_H,start_x:start_x+output_W]
        return (img, target)
    return crop

@joint_transforms_registry.register
def joint_keep_ratio_resize(transforms_cfg):
    output_H, output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    assert output_H == output_W
    size = (output_H, output_W)
    def crop(img, target):
        assert img.shape[-2:] == target.shape[-2:]
        img_H = img.shape[-2]
        img_W = img.shape[-1]
        scale = output_W / float(max(img_H, img_W))
        target_H, target_W = int(img_H * scale), int(img_W * scale)
        img = tr_F.resize(img, (target_H, target_W))
        # place on zero tensors
        # img_bg = torch.zeros(img.shape[:-2] + size, dtype = img.dtype, device = img.device)
        # img_bg[:, 0:target_H, 0:target_W] = img
        if len(target.shape) == 2:
            # HxW?
            target = target.reshape((1,) + target.shape)
            target = tr_F.resize(target, (target_H, target_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            target = target.reshape(target.shape[1:])
            # target_bg = torch.zeros(size, dtype = target.dtype, device = target.device)
            # target_bg[:target_H, :target_W] = target
        else:
            assert len(target.shape) == 3
            target = tr_F.resize(target, (target_H, target_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            # target_bg = torch.zeros(target.shape[:-2] + size, dtype = target.dtype, device = target.device)
            # target_bg[:, :target_H, :target_W] = target
        return (img, target)
    return crop

@joint_transforms_registry.register
def joint_center_crop(transforms_cfg):
    output_H, output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    size = (output_H, output_W)
    def crop(img, target):
        assert img.shape[-2:] == target.shape[-2:]
        img = tr_F.center_crop(img, size)
        target = tr_F.center_crop(img, size)
        return (img, target)
    return crop

@joint_transforms_registry.register
def joint_random_horizontal_flip(transforms_cfg):
    # The default setting is p=0.5 in torchvision
    # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip
    def crop(img, target):
        if torch.rand(1) < 0.5:
            img = tr_F.hflip(img)
            target = tr_F.hflip(target)
        return (img, target)
    return crop
