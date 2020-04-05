import torch
from torchvision import datasets, transforms

def get_train_set(cfg):
    return datasets.ImageNet('~/datasets', split = 'train', download = True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

def get_val_set(cfg):
    return datasets.ImageNet('~/datasets', split = 'val', download = True,
                        transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))