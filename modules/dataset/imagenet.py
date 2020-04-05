import torch
from torchvision import datasets, transforms
from .baseset import base_set

def get_train_set(cfg):
    return datasets.ImageNet('~/datasets', split = 'train', download = True,
                       transform = transforms.ToTensor())

def get_val_set(cfg):
    return datasets.ImageNet('~/datasets', split = 'val', download = True,
                        transform = transforms.ToTensor())