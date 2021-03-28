import sys
import os
import torch
from torchvision import datasets, transforms
from .baseset import base_set

download_ds = False if os.path.exists('/data/mnist') else True

def get_train_set(cfg):
    ds = datasets.MNIST('/data/mnist', train = True, download = download_ds,
                       transform = transforms.ToTensor())
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =  datasets.MNIST('/data/mnist', train = False, download = download_ds,
                        transform = transforms.ToTensor())
    return base_set(ds, "test", cfg)