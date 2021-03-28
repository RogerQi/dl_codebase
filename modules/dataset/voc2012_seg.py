import sys
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from .baseset import base_set

toTensorFunc = transforms.ToTensor()
download_sbd = False if os.path.exists('/data/sbd') else True
download_voc = False if os.path.exists('/data/VOCdevkit') else True

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
    ds = sbd_seg('/data/sbd', image_set='train_noval', mode="segmentation", download=download_sbd)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =  voc2012_seg('/data/', image_set='val', download = download_voc, transform=transforms.ToTensor())
    return base_set(ds, "test", cfg)