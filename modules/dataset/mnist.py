import torch
from torchvision import datasets, transforms

def get_train_set(cfg):
    return datasets.MNIST('../data', train = True, download = True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

def get_val_set(cfg):
    pass

def get_test_set(cfg):
    return datasets.MNIST('../data', train=False, download = True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))