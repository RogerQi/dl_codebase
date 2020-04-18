import torch
from torchvision import datasets, transforms
from .baseset import base_set

def get_train_set(cfg):
    ds = datasets.CIFAR10('../datasets', train = True, download = True,
                       transform = transforms.ToTensor())
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =  datasets.CIFAR10('../datasets', train = False, download = True,
                        transform = transforms.ToTensor())
    return base_set(ds, "test", cfg)