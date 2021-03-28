import numpy as np
import torch
from torchvision import datasets, transforms
from .baseset import base_set

toTensorFunc = transforms.ToTensor()

class voc2012_seg(datasets.VOCSegmentation):
    def __getitem__(self, idx):
        img, label_pil = datasets.VOCSegmentation.__getitem__(self, idx)
        label_np = np.array(label_pil)
        # Only 20 classes in VOC2012. Those pixels whose class is
        # greater than 20 are edges
        label_np[label_np > 20] = 0
        return img, torch.tensor(label_np)

class sbd_seg(datasets.SBDataset):
    def __getitem__(self, idx):
        img, label_pil = datasets.SBDataset.__getitem__(self, idx)
        img = toTensorFunc(img)
        label_np = np.array(label_pil, dtype=np.long)
        return img, torch.tensor(label_np)

def get_train_set(cfg):
    ds = sbd_seg('/data/', image_set='train_noval', mode="segmentation", download=True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =  voc2012_seg('/data/', image_set='val', download = True, transform=transforms.ToTensor())
    return base_set(ds, "test", cfg)