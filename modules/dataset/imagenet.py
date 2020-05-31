import torch
from torchvision import datasets, transforms
from .baseset import base_set

def get_train_set(cfg):
    ds = datasets.ImageNet('/data', split = 'train',
                       transform = transforms.ToTensor())
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = datasets.ImageNet('/data', split = 'val',
                        transform = transforms.ToTensor())
    return base_set(ds, "test", cfg)